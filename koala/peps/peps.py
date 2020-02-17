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
from . import contraction, update, sites


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
        env = contraction.create_env_cache(self)
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            rows = [site // self.ncol for site in sites]
            up, down = min(rows), max(rows)
            e += np.real_if_close(contraction.inner_with_env(
                other[up:down+1].dagger().apply(self[up:down+1]),
                env, up, down
            ))
        return e

    def contract(self, approach='MPS', **svdargs):
        return contraction.contract(self, approach='MPS', **svdargs)

    def inner(self, other):
        return self.dagger().apply(other).contract()

    def statevector(self):
        from .. import statevector
        return statevector.StateVector(self.contract(), self.backend)

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
            grid[idx] = sites.contract_z(a, b)
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
                tn[idx] = sites.flip_z(tsr)
            else:
                tn[idx] = tsr.copy()
        return PEPS(tn, self.backend)

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
            tn[idx] = sites.rotate_z(tsr, num_rotate90)
        return PEPS(tn, self.backend)


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
