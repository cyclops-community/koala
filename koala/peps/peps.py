"""
This module defines PEPS and operations on it.
"""

import random
from math import sqrt
from numbers import Number
from string import ascii_letters as chars

import numpy as np
import tensorbackends

from ..quantum_state import QuantumState
from ..gates import tensorize
from . import contract, update


class PEPS(QuantumState):
    def __init__(self, grid, backend):
        self.backend = tensorbackends.get(backend)
        self.grid = grid

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def nsite(self):
        return self.nrow * self.ncol

    def __getitem__(self, position):
        item = self.grid[position]
        if isinstance(item, np.ndarray):
            if item.ndim == 1:
                if isinstance(position[0], int):
                    item = item.reshape(1, -1)
                else:
                    item = item.reshape(-1, 1)
            return PEPS(item, self.backend)
        return item

    def copy(self):
        grid = np.empty_like(self.grid)
        for idx, tensor in np.ndenumerate(self.grid):
            grid[idx] = tensor.copy()
        return PEPS(grid, self.backend)

    def conjugate(self):
        grid = np.empty_like(self.grid)
        for idx, tensor in np.ndenumerate(self.grid):
            grid[idx] = tensor.conj()
        return PEPS(grid, self.backend)

    def apply_gate(self, gate, threshold=None, maxrank=None, randomized_svd=False):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        self.apply_operator(tensor, gate.qubits, threshold=threshold, maxrank=maxrank, randomized_svd=randomized_svd)

    def apply_circuit(self, gates, threshold=None, maxrank=None, randomized_svd=False):
        for gate in gates:
            self.apply_gate(gate, threshold=threshold, maxrank=maxrank, randomized_svd=randomized_svd)

    def apply_operator(self, operator, sites, threshold=None, maxrank=None, randomized_svd=False):
        operator = self.backend.astensor(operator)
        positions = [divmod(site, self.ncol) for site in sites]
        if len(positions) == 1:
            update.apply_single_site_operator(self, operator, positions[0])
        elif len(positions) == 2 and is_two_local(*positions):
            update.apply_local_pair_operator(self, operator, positions, threshold, maxrank, randomized_svd)
        else:
            raise ValueError('nonlocal operator is not supported')

    def __add__(self, other):
        if isinstance(other, PEPS) and self.backend == other.backend:
            return self.add(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, PEPS) and self.backend == other.backend:
            return self.add(other, coeff=-1.0)
        else:
            return NotImplemented

    def __imul__(self, a):
        if isinstance(a, Number):
            multiplier = a ** (1/(self.nrow * self.ncol))
            for idx in np.ndindex(*self.shape):
                self.grid[idx] *= multiplier
            return self
        else:
            return NotImplemented

    def __itruediv__(self, a):
        if isinstance(a, Number):
            divider = a ** (1/(self.nrow * self.ncol))
            for idx in np.ndindex(*self.shape):
                self.grid[idx] /= divider
            return self
        else:
            return NotImplemented

    def norm(self):
        return sqrt(np.real_if_close(self.inner(self)))

    def add(self, other, *, coeff=1.0):
        """
        Add two PEPS of the same grid shape and return the sum as a third PEPS also with the same grid shape.
        """
        if self.shape != other.shape:
            raise ValueError(f'PEPS shapes do not match: {self.shape} != {other.shape}')
        grid = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(*self.grid.shape):
            internal_bonds = []
            external_bonds = [4, 5]
            (external_bonds if i == 0 else internal_bonds).append(0)
            (external_bonds if j == self.shape[1] - 1 else internal_bonds).append(1)
            (external_bonds if i == self.shape[0] - 1 else internal_bonds).append(2)
            (external_bonds if j == 0 else internal_bonds).append(3)
            grid[i, j] = tn_add(self.backend, self[i, j], other[i, j], internal_bonds, external_bonds, 1, coeff)
        return PEPS(grid, self.backend)

    def amplitude(self, indices):
        if len(indices) != self.nsite:
            raise ValueError('indices number and sites number do not match')
        indices = np.array(indices).reshape(*self.shape)
        grid = np.empty_like(self.grid, dtype=object)
        zero = self.backend.astensor(np.array([1,0], dtype=complex).reshape(2, 1))
        one = self.backend.astensor(np.array([0,1], dtype=complex).reshape(2, 1))
        for idx, tensor in np.ndenumerate(self.grid):
            grid[idx] = self.backend.einsum('ijklxp,xq->ijklpq', tensor, one if indices[idx] else zero)
        return PEPS(grid, self.backend).contract()

    def probability(self, indices):
        return np.abs(self.amplitude(indices))**2

    def expectation(self, observable, use_cache=False):
        if use_cache:
            return self._expectation_with_cache(observable)
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            e += np.real_if_close(self.inner(other))
        return e

    def _expectation_with_cache(self, observable):
        env = self.create_env_cache()
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            rows = [site // self.ncol for site in sites]
            up, down = min(rows), max(rows)
            e += np.real_if_close(other[up:down+1].inner_PEPS(self[up:down+1]).inner_with_env(env, up, down))
        return e

    # def contract(self):
    #     return contract.to_statevector(self)

    def inner(self, other):
        return self.inner_PEPS(other).contract()

    def statevector(self):
        from .. import statevector
        return statevector.StateVector(self.contract(), self.backend)

    
    def create_env_cache(self):
        peps_obj = self.norm_PEPS()
        _up, _down = {}, {}
        for i in range(peps_obj.shape[0]):
            _up[i] = peps_obj[:i].contract_to_MPS() if i != 0 else None
            _down[i] = peps_obj[i+1:].contract_to_MPS() if i != self.nrow - 1 else None
        return _up, _down

    def inner_with_env(self, env, up_idx, down_idx):
        up, down = env[0][up_idx], env[1][down_idx]
        if up is None and down is None:
            peps_obj = self
        elif up is None:
            peps_obj = self.concatenate(down)
        elif down is None:
            peps_obj = up.concatenate(self)
        else:
            peps_obj = up.concatenate(self).concatenate(down)
        return peps_obj.contract()


    def apply(self, other):
        """
        Apply a PEPS/PEPO to another PEPS/PEPO. Only the first pair of physical indices is contracted; the other physical indices are left in the order of A, B.

        Parameters
        ----------
        other: PEPS
            The second PEPS/PEPO.

        Returns
        -------
        output: PEPS
            The PEPS generated by the application.
        """
        grid = np.empty_like(self.grid)
        for (idx, a), b in zip(np.ndenumerate(self.grid), other.grid.flat):
            grid[idx] = self._tensor_dot(a, b, 'z')
        return PEPS(grid, self.backend)

    def concatenate(self, other, axis=0):
        """
        Concatenate two PEPS along the given axis.
        
        Parameters
        ----------
        other: PEPS
            The second PEPS

        axis: int, optional
            The axis along which the PEPS will be concatenated.

        Returns
        -------
        output: PEPS
            The concatenated PEPS.
        """
        return PEPS(np.concatenate((self.grid, other.grid), axis), self.backend)

    # def conjugate(self):
    #     """
    #     Compute the conjugate of the PEPS.

    #     Returns
    #     -------
    #     output: PEPS
    #         The conjugated PEPS.
    #     """
    #     tn = np.empty_like(self.grid)
    #     for idx, tsr in np.ndenumerate(self.grid):
    #         tn[idx] = tsr.conj()
    #     return PEPS(tn, self.backend)

    def contract(self, approach='MPS', **svdargs):
        """
        Contract the PEPS to a single tensor or a scalar(a \"0-tensor\").

        Parameters
        ----------
        approach: str, optional
            The approach to contract.

        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: self.backend.tensor or scalar
            The contraction result.
        """
        approach = approach.lower()
        if approach in ['mps', 'bmps', 'bondary']:
            return self.contract_BMPS(**svdargs)
        elif approach in ['meshgrid', 'square', 'squares']:
            return self.contract_squares(**svdargs)
        elif approach in ['trg', 'terg']:
            return self.contract_TRG(**svdargs)
        elif approach in ['s', 'snake', 'snakes']:
            return self.contract_snake()

    def contract_BMPS(self, mps_mult_mpo=None, **svdargs):
        """
        Contract the PEPS by contracting each MPS layer.
        
        Parameters
        ----------
        mps_mult_mpo: method or None, optional
            The method used to apply an MPS to another MPS/MPO.

        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: self.backend.tensor or scalar
            The contraction result.
        """
        # contract boundary MPS down
        mps = self.contract_to_MPS(False, mps_mult_mpo, **svdargs).grid.reshape(-1)
        # contract the last MPS to a single tensor
        result = mps[0]
        for tsr in mps[1:]:
            result = self._tensor_dot(result, tsr, 'y')
        return result.item() if result.size == 1 else result.reshape(
            *[int(result.size ** (1 / self.grid.size))] * self.grid.size
            ).transpose(*[i + j * self.nrow for i, j in np.ndindex(*self.shape)])

    def contract_env(self, row_range, col_range, **svdargs):
        """
        Contract the surrounding environment to four MPS around the core sites.
        
        Parameters
        ----------
        row_range: tuple or int
            A two-int tuple specifying the row range of the core sites, i.e. [row_range[0] : row_range[1]]. 
            If only an int is given, it is equivalent to (row_range, row_range+1).

        col_range: tuple or int
            A two-int tuple specifying the column range of the core sites, i.e. [:, col_range[0] : col_range[1]]. 
            If only an int is given, it is equivalent to (col_range, col_range+1).

        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: PEPS
            The new PEPS consisting of core sites and contracted environment.
        """
        if isinstance(row_range, int):
            row_range = (row_range, row_range+1)
        if isinstance(col_range, int):
            col_range = (col_range, col_range+1)
        mid_peps = self[row_range[0]:row_range[1]].copy()
        if row_range[0] > 0:
            mid_peps = self[:row_range[0]].contract_to_MPS(**svdargs).concatenate(mid_peps)
        if row_range[1] < self.nrow:
            mid_peps = mid_peps.concatenate(self[row_range[1]:].contract_to_MPS(**svdargs))
        env_peps = mid_peps[:,col_range[0]:col_range[1]]
        if col_range[0] > 0: 
            env_peps = mid_peps[:,:col_range[0]].contract_to_MPS(horizontal=True, **svdargs).concatenate(env_peps, axis=1)
        if col_range[1] < mid_peps.shape[1]:
            env_peps = env_peps.concatenate(mid_peps[:,col_range[1]:].contract_to_MPS(horizontal=True, **svdargs), axis=1)
        return env_peps

    def contract_snake(self):
        """
        Contract the PEPS by contracting sites in the row-major order.

        Returns
        -------
        output: self.backend.tensor or scalar
            The contraction result.

        References
        ----------
        https://arxiv.org/pdf/1905.08394.pdf
        """
        head = self.grid[0,0]
        for i, mps in enumerate(self.grid):
            for tsr in mps[int(i==0):]:
                head = self.backend.einsum('agbcdef->a(gb)cdef', 
                head.reshape((head.shape[0] // tsr.shape[0], tsr.shape[0]) + head.shape[1:]))
                tsr = self.backend.einsum('agbcdef->abc(gd)ef', tsr.reshape((1,) + tsr.shape))
                head = self._tensor_dot(head, tsr, 'y')
            head = head.transpose(2, 1, 0, 3, 4, 5)
        return head.item() if head.size == 1 else head.reshape(*[int(head.size ** (1 / self.grid.size))] * self.grid.size)

    def contract_squares(self, **svdargs):
        """
        Contract the PEPS by contracting two neighboring tensors to one recursively. 
        The neighboring relationship alternates from horizontal and vertical.
        
        Parameters
        ----------
        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: self.backend.tensor or scalar
            The contraction result.
        """
        tn = self.grid
        new_tn = np.empty((int((self.nrow + 1) / 2), self.ncol), dtype=object)
        for ((i, j), a), b in zip(np.ndenumerate(tn[:-1:2,:]), tn[1::2,:].flat):
            new_tn[i,j] = self._tensor_dot(a, b, 'x')
            if svdargs and j > 0 and new_tn.shape != (1, 2):
                new_tn[i,j-1], new_tn[i,j] = self._tensor_dot(new_tn[i,j-1], new_tn[i,j], 'y', **svdargs)
        # append the left edge if nrow/ncol is odd
        if self.nrow % 2 == 1:
            for i, a in enumerate(tn[-1]):
                new_tn[-1,i] = a.copy()
        # base case
        if new_tn.shape == (1, 1):
            return new_tn[0,0].item() if new_tn[0,0].size == 1 else new_tn[0,0]
        # alternate the neighboring relationship and contract recursively
        return PEPS(new_tn, self.backend).rotate().contract_squares(**svdargs)

    def contract_to_MPS(self, horizontal=False, mps_mult_mpo=None, **svdargs):
        """
        Contract the PEPS to an MPS.
        
        Parameters
        ----------
        horizontal: bool, optional
            Control whether to contract from top to bottom or from left to right. Will affect the output MPS direction.
        
        mps_mult_mpo: method or None, optional
            The method used to apply an MPS to another MPS/MPO.

        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: PEPS
            The resulting MPS (as a `PEPS` object of shape `(1, N)` or `(M, 1)`).
        """
        if mps_mult_mpo is None:
            mps_mult_mpo = self._mps_mult_mpo
        if horizontal:
            self.rotate(-1)
        mps = self.grid[0]
        for mpo in self.grid[1:]:
            mps = mps_mult_mpo(mps, mpo, **svdargs)
        mps = mps.reshape(1, -1)
        p = PEPS(mps, self.backend)
        return p.rotate() if horizontal else p

    def contract_TRG(self, **svdargs):
        """
        Contract the PEPS using Tensor Renormalization Group.
        
        Parameters
        ----------
        svdargs: dict, optional
            Arguments for SVD truncation. Will perform SVD if given. See `tensorview.einsvd`.

        Returns
        -------
        output: self.backend.tensor or scalar
            The contraction result.

        References
        ----------
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.120601
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.78.205116
        """
        # base case
        if self.shape <= (2, 2):
            return self.contract_BMPS(**svdargs)
        if not svdargs:
            svdargs = {'rank': None}
        # SVD each tensor into two
        tn = np.empty(self.shape + (2,), dtype=object)
        for (i, j), tsr in np.ndenumerate(self.grid):
            tn[i,j,0], tn[i,j,1] = tbs.einsvd('abcdpq->abi,icdpq' if (i+j) % 2 == 0 else 'abcdpq->aidpq,bci', tsr)
            tn[i,j,(i+j)%2] = tn[i,j,(i+j)%2].reshape(tn[i,j,(i+j)%2].shape + (1, 1))
        return self._contract_TRG(tn, **svdargs)

    # def copy(self):
    #     """
    #     Return a deep copy of the PEPS.

    #     Returns
    #     -------
    #     output: PEPS
    #     """
    #     tn = np.empty_like(self)
    #     for idx, tsr in np.ndenumerate(self):
    #         tn[idx] = tsr.coy()
    #     return PEPS(tn, self.backend)

    def dagger(self):
        """
        Compute the Hermitian conjugate of the PEPS. Equivalent to take `conjugate` then `flip`.

        Returns
        -------
        output: PEPS
        """
        return self.conjugate().flip()

    def flip(self, *indices):
        """
        Flip the direction of physical indices for specified sites.
        
        Parameters
        ----------
        indices: iterable, optional
            Indices of sites (tensors) to flip. Specify as `(i, j)` or `((i1, j1), (i2, j2), ...)`, where `i` and `j` should be int.
            Will flip all sites if left as `None`.
        
        Returns
        -------
        output: PEPS
        """
        if indices and isinstance(indices[0], int):
            indices = (indices, )
        tn = np.empty_like(self.grid)
        for idx, tsr in np.ndenumerate(self.grid):
            if not indices or idx in indices:
                tn[idx] = tsr.transpose(0, 1, 2, 3, 5, 4)
            else:
                tn[idx] = tsr.copy()
        return PEPS(tn, self.backend)

    def inner_PEPS(self, other):
        """
        Compute the inner product of two PEPS, i.e. <psi|phi>. Equivalent to self apply on B^dagger.
        
        Parameters
        ----------
        other: PEPS
            The second PEPS.
        
        Returns
        -------
        output: PEPS
        """
        return self.dagger().apply(other)

    def norm_PEPS(self):
        """
        Compute the PEPS contracted with its Hermitian conjugate, i.e. <psi|psi>.
        
        Returns
        -------
        output: PEPS
        """
        return self.inner_PEPS(self)

    def rotate(self, num_rotate90=1):
        """
        Rotate the PEPS counter-clockwise by 90 degrees * the specified times. Will cause the tensors to transpose accordingly.

        Parameters
        ----------
        num_rotate90: int, optional
            Number of 90 degree rotations.

        Returns
        -------
        output: PEPS
        """
        tn = self.grid
        for _ in range(num_rotate90 % 4):
            tn = np.rot90(tn)
            for idx, tsr in np.ndenumerate(tn):
                tn[idx] = tsr.transpose(1, 2, 3, 0, 4, 5)
        return PEPS(tn, self.backend)

    # def switch_backend(self, backend):
    #     """
    #     Switch the backend of the PEPS. Will cause every tensor to change its type.

    #     Parameters
    #     ----------
    #     num_rotate90: int, optional
    #         Number of 90 degree rotations.
            
    #     Returns
    #     -------
    #     output: PEPS
    #     """
    #     return PEPS(self, backend=backend, copy=True)

    # def transpose(self):
    #     """
    #     Transpose the PEPS. Will cause the tensors to transpose accordingly.

    #     Returns
    #     -------
    #     output: PEPS
    #     """
    #     tn = self.grid.T
    #     for idx, tsr in np.ndenumerate(tn):
    #         tn[idx] = tsr.transpose(3, 2, 1, 0, 4, 5)
    #     return PEPS(tn, self.backend)

    def _contract_TRG(self, tn, **svdargs):
        # base case
        if tn.shape == (2, 2, 2):
            p = np.empty((2, 2), dtype=object)
            for i, j in np.ndindex((2, 2)):
                p[i,j] = self.backend.einsum('abipq,icdPQ->abcdp+Pq+Q' if (i+j) % 2 == 0 else 'aidpq,bciPQ->abcdp+Pq+Q', tn[i,j][0], tn[i,j][1])
            return PEPS(p, self.backend).contract_BMPS()
        
        # contract specific horizontal and vertical bonds and SVD truncate the generated squared bonds
        for i, j in np.ndindex(tn.shape[:2]):
            if j > 0 and j % 2 == 0:
                k = 1 - i % 2
                l = j - ((i // 2 * 2 + j) % 4 == 0)
                tn[i,l][k] = self.backend.einsum('ibapq,ABiPQ->Ab+Bap+Pq+Q' if k else 'biapq,BAiPQ->b+BAap+Pq+Q', tn[i,j-1][k], tn[i,j][k])
                if i % 2 == 1:
                    tn[i-1,l][1], tn[i,l][0] = self.backend.einsum('aidpq,iBCPQ->aBCdp+Pq+Q', tn[i-1,l][1], tn[i,l][0], **svdargs)
            if i > 0 and i % 2 == 0:
                k = 1 - j % 2
                l = int((i + j // 2 * 2) % 4 == 0)
                tn[i-l,j][l] = self.backend.einsum('biapq,iBAPQ->b+BAap+Pq+Q' if k else 'aibpq,iABPQ->aAb+Bp+Pq+Q', tn[i-1,j][1], tn[i,j][0])
                if j % 2 == 1:
                    tn[i-l,j-1][l], tn[i-l,j][l] = self.backend.einsum('icdpq,ABiPQ->ABcdp+Pq+Q', tn[i-l,j-1][l], tn[i-l,j][l], **svdargs)
        
        # contract specific diagonal bonds and generate a smaller tensor network
        new_tn = np.empty((tn.shape[0] // 2 + 1, tn.shape[1] // 2 + 1, 2), dtype=object)
        for i, j in np.ndindex(tn.shape[:2]):
            m, n = (i + 1) // 2, (j + 1) // 2
            if (i + j) % 4 == 2 and i % 2 == 0:
                if tn[i,j][0] is None:
                    new_tn[m,n][1] = tn[i,j][1]
                elif tn[i,j][1] is None:
                    new_tn[m,n][1] = tn[i,j][0]
                else:
                    new_tn[m,n][1] = self.backend.einsum('abipq,iCAPQ->bCa+Ap+Pq+Q' if i == 0 else 'aibpq,iCBPQ->aCb+Bp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 0 and i % 2 == 1:
                new_tn[m,n][0] = self.backend.einsum('abipq,iBCPQ->ab+BCp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 3 and i % 2 == 0:
                new_tn[m,n][1] = self.backend.einsum('aibpq,ACiPQ->a+ACbp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 3 and i % 2 == 1:
                new_tn[m,n][0] = self.backend.einsum('aibpq,CBiPQ->aCb+Bp+Pq+Q', tn[i,j][0], tn[i,j][1])
            else:
                if new_tn[m,n][0] is None:
                    new_tn[m,n][0] = tn[i,j][0]
                if new_tn[m,n][1] is None:
                    new_tn[m,n][1] = tn[i,j][1]
        
        # SVD truncate the squared bonds generated by the diagonal contractions
        for i, j in np.ndindex(new_tn.shape[:2]):
            if (i + j) % 2 == 0 and new_tn[i,j][0] is not None and new_tn[i,j][1] is not None:
                new_tn[i,j][0], new_tn[i,j][1] = self.backend.einsum('abipq,iCDPQ->abCDp+Pq+Q', new_tn[i,j][0], new_tn[i,j][1], **svdargs)
            elif (i + j) % 2 == 1:
                new_tn[i,j][0], new_tn[i,j][1] = self.backend.einsum('aidpq,BCiPQ->aBCdp+Pq+Q', new_tn[i,j][0], new_tn[i,j][1], **svdargs)

        return self._contract_TRG(new_tn, **svdargs)

    def _mps_mult_mpo(self, mps, mpo, **svdargs):
        # if mpo[0].shape[2] == 1:
            # svdargs = {}
        new_mps = np.empty_like(mps)
        for i, (s, o) in enumerate(zip(mps, mpo)):
            new_mps[i] = self._tensor_dot(s, o, 'x')
            if svdargs and i > 0:
                new_mps[i-1], new_mps[i] = self._tensor_dot(new_mps[i-1], new_mps[i], 'y', **svdargs)
        return new_mps

    def _tensor_dot(self, a, b, axis, **svdargs):
        axis= axis.lower()
        if axis == 'x':
            einstr = 'abidpq,iBcDPQ->a(bB)c(dD)(pP)(qQ)'
        elif axis == 'y':
            einstr = 'aicdpq,AbCiPQ->(aA)b(cC)d(pP)(qQ)'
        elif axis == 'z':
            einstr = 'abcdpi,ABCDiq->(aA)(bB)(cC)(dD)pq'
        return self.backend.einsum(einstr, a, b, **svdargs)



def tn_add(backend, a, b, internal_bonds, external_bonds, coeff_a, coeff_b):
    """
    Helper function for addition of two tensor network states with the same structure. 
    Add two site from two tensor network states respecting specified inner and external bond structure. 
    """
    ndim = a.ndim
    shape_a = np.array(np.shape(a))
    shape_b = np.array(np.shape(b))
    shape_c = np.copy(shape_a)
    shape_c[internal_bonds] += shape_b[internal_bonds]
    lim = np.copy(shape_a).astype(object)
    lim[external_bonds] = None    
    a_ind = tuple([slice(lim[i]) for i in range(ndim)])
    b_ind = tuple([slice(lim[i], None) for i in range(ndim)])
    c = backend.zeros(shape_c, dtype=a.dtype)
    c[a_ind] += a * coeff_a
    c[b_ind] += b * coeff_b
    return c


def is_two_local(p, q):
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    return dx == 1 and dy == 0 or dx == 0 and dy == 1
