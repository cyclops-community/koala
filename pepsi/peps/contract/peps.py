# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np

import tensorbackends as tbs


class PEPS(object):
    """
    Initialized by a 2d array or list containing `numpy.ndarray` or `ctf.tensor`. One may also provide a customized backend (see `tensorview/backends`).
    The tensor network is stored as a 2d array of `self._backend.tensor` objects.
    The order of indices is clockwise from 12 o'clock on the horizontal surface then top and bottom for physical indices (if any).
        4 0
        |/
    3---o---1
       /|
      2 5

    Attributes
    ----------
    backend: str
        Module name of the current backend. For the actual backend module, use PEPS._backend.

    dims: numpy.ndarray
        The actual stored dimensions(shapes) for each tensor.

    nphyidx: numpy.ndarray
        Number of physical indices for each tensor.

    size: int
        The total number of elements in the PEPS.

    shape: tuple
        A length 2 tuple containing the height and width of the PEPS.
    
    type: str
        The type of the PEPS. "open", "infinite", or "periodic".

    vdims: numpy.ndarray
        The virtual dimensions(shapes) for each tensor.

    Methods
    -------
    apply:
        Apply a PEPS/PEPO to another PEPS/PEPO.

    contract:
        Contract the PEPS by the specified approach.

    contract_BMPS:
        Contract a PEPS by contracting each MPS layer.

    contract_BMPS:
        Contract a PEPS by contracting each MPS layer.

    contract_squares:
        Contract the PEPS by contracting two neighboring tensors to one recursively. 
        The neighboring relationship alternates from horizontal and vertical.

    contract_TRG:
        Contract the PEPS using Tensor Renormalization Group.

    copy:
        Return a deep copy of the PEPS.

    norm:
        Compute the PEPS contracted with its Hermitian conjugate, i.e. <phi|phi>.

    rotate:
        Rotate the PEPS counter-clockwise by 90 degrees * the specified times. Will cause tensors to transpose accordingly.

    switch_backend:
        Switch the backend of the PEPS. Will cause every tensor to change its type.

    transpose:
        Transpose the PEPS. Will cause tensors to transpose accordingly.
    """
            
    def __init__(self, tensors, PEPS_like=None, backend=None):
        transpose = True
        if isinstance(tensors, PEPS):
            PEPS_like = PEPS_like or tensors
        if PEPS_like is not None:
            backend = backend or PEPS_like._backend
            transpose = False
        self._backend = tbs.get(backend or 'numpy')

        self._tn = np.empty(tensors.shape, dtype=object)
        for idx, tsr in np.ndenumerate(tensors):
            # self._tn[i,j] = self._backend.tensor(tensors[i,j], None, pseudo_idx, backend=backend, copy=copy)
            # self._tn[i,j].insert_pseudo(*range(self._tn[i,j].vndim, 6))
            self._tn[idx] = tsr.transpose(*((0, 3, 2, 1) + tuple(range(4, tsr.ndim)))) if transpose else tsr.copy()
            self._tn[idx] = self._tn[idx].reshape(*(self._tn[idx].shape + tuple([1] * (6 - tsr.ndim))))


    def __abs__(self):
        new_tn = np.empty_like(self._tn)
        for idx, tsr in np.ndenumerate(self._tn):
            new_tn[idx] = abs(tsr)
        return PEPS(new_tn, self)

    def __add__(self, other):
        if np.isscalar(other):
            new_tn = np.empty_like(self._tn)
            for idx, tsr in np.ndenumerate(self._tn):
                new_tn[idx] = tsr + other
            return PEPS(new_tn, self)
        if isinstance(other, PEPS):
            if self.shape != other.shape:
                raise ValueError("PEPS: addends need to have the same shape")
            new_tn = np.empty_like(self._tn)
            for idx, tsr in np.ndenumerate(self._tn):
                new_tn[idx] = tsr + other._tn[idx]
            return PEPS(new_tn, self)
        raise NotImplementedError()

    def __contains__(self, item):
        return item in self._tn

    def __getitem__(self, key):
        item = self._tn[key]
        if isinstance(item, np.ndarray):
            if item.ndim == 1:
                if isinstance(key[0], int):
                    item = item.reshape(1, -1)
                else:
                    item = item.reshape(-1, 1)
            return PEPS(item, self)
        return item

    def __getattr__(self, attr):
        return getattr(self._tn, attr)

    def __iter__(self):
        return self._tn.__iter__()

    def __len__(self):
        return len(self._tn)

    def __mul__(self, other):
        if np.isscalar(other):
            new_tn = np.empty_like(self._tn)
            for idx, tsr in np.ndenumerate(self._tn):
                new_tn[idx] = tsr * other
            return PEPS(new_tn, self)
        raise NotImplementedError()

    def __neg__(self):
        new_tn = np.empty_like(self._tn)
        for idx, tsr in np.ndenumerate(self._tn):
            new_tn[idx] = -tsr
        return PEPS(new_tn, self)

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return repr(self._tn)

    def __rmul__(self, other):
        return self * other

    def __setitem__(self, key, value):
        if type(self._tn[key]) is type(value):
            self._tn[key] = value
        else:
            raise TypeError("PEPS: new and old value should have the same type")

    def __sub__(self, other):
        if np.isscalar(other):
            new_tn = np.empty_like(self._tn)
            for idx, tsr in np.ndenumerate(self._tn):
                new_tn[idx] = tsr - other
            return PEPS(new_tn, self)
        if isinstance(other, PEPS):
            if self.shape != other.shape:
                raise ValueError("PEPS: subtrahend and minuend need to have the same shape")
            new_tn = np.empty_like(self._tn)
            for idx, tsr in np.ndenumerate(self._tn):
                new_tn[idx] = tsr - other._tn[idx]
            return PEPS(new_tn, self)
        raise NotImplementedError()

    def __str__(self):
        return str(self._tn)

    @property
    def backend(self):
        return self._backend.name

    # @backend.setter
    # def backend(self, backend):
    #     self._backend = tbs.get(backend)
    #     for idx, tsr in np.ndenumerate(self._tn):
    #         self._tn[idx] = self._backend.tensor(backend.asarray(tsr))

    # @property
    # def nphyidx(self):
    #     nphyidx = np.empty_like(self._tn, dtype=int)
    #     for idx, tsr in np.ndenumerate(self._tn):
    #         nphyidx[i,j] = sum(1 for i in tsr.metalen()[4:] if i != 0)
    #     return nphyidx

    @property
    def dims(self):
        dims = np.empty_like(self._tn, dtype=tuple)
        for (i, j), tsr in np.ndenumerate(self._tn):
            dims[i,j] = tsr.shape
        return dims

    @property
    def nrow(self):
        return self.shape[0]

    @property
    def ncol(self):
        return self.shape[1]

    @property
    def nsite(self):
        return self.nrow * self.ncol

    # @property
    # def rdims(self):
    #     dims = np.empty_like(self._tn, dtype=tuple)
    #     for (i, j), tsr in np.ndenumerate(self._tn):
    #         dims[i,j] = tsr.rshape
    #     return dims

    @property
    def size(self):
        sum = 0
        for tsr in self._tn.flat:
            sum += tsr.size
        return sum

    @property
    def shape(self):
        return self._tn.shape

    # @property
    # def vdims(self):
    #     dims = np.empty_like(self._tn, dtype=tuple)
    #     for (i, j), tsr in np.ndenumerate(self._tn):
    #         dims[i,j] = tsr.vshape
    #     return dims

    def apply(self, B):
        """
        Apply a PEPS/PEPO to another PEPS/PEPO. Only the first pair of physical indices is contracted; the other physical indices are left in the order of A, B.

        Parameters
        ----------
        B: PEPS
            The second PEPS/PEPO.

        Returns
        -------
        output: PEPS
            The PEPS generated by the application.
        """
        A = self._tn
        B = B._tn
        tn = np.empty_like(A)
        for (idx, a), b in zip(np.ndenumerate(A), B.flat):
            tn[idx] = self._tensor_dot(a, b, 'z')
        return PEPS(tn, self)

    def concatenate(self, B, axis=0):
        """
        Concatenate two PEPS along the given axis.
        
        Parameters
        ----------
        B: PEPS
            The second PEPS

        axis: int, optional
            The axis along which the PEPS will be concatenated.

        Returns
        -------
        output: PEPS
            The concatenated PEPS.
        """
        return PEPS(np.concatenate((self._tn, B._tn), axis), self)

    def conjugate(self):
        """
        Compute the conjugate of the PEPS.

        Returns
        -------
        output: PEPS
            The conjugated PEPS.
        """
        tn = np.empty_like(self._tn)
        for idx, tsr in np.ndenumerate(self._tn):
            tn[idx] = tsr.conj()
        return PEPS(tn, self)

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
        output: self._backend.tensor or scalar
            The contraction result.
        """
        approach = approach.lower()
        if approach in ['mps', 'bmps', 'bondary']:
            return self.contract_BMPS(**svdargs)
        elif approach in ['meshgrid', 'square', 'squares']:
            return self.contract_squares(**svdargs)
        elif approach in ['trg', 'terg']:
            return self.contract_TRG(**svdargs)

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
        output: self._backend.tensor or scalar
            The contraction result.
        """
        # contract boundary MPS down
        mps = self.contract_to_MPS(False, mps_mult_mpo, **svdargs)._tn.reshape(-1)
        # contract the last MPS to a single tensor
        result = mps[0]
        for tsr in mps[1:]:
            result = self._tensor_dot(result, tsr, 'y')
        return result.item() if result.size == 1 else result

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
        output: self._backend.tensor or scalar
            The contraction result.
        """
        tn = self._tn
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
        return PEPS(new_tn, self).rotate().contract_squares(**svdargs)

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
        mps = self._tn[0]
        for mpo in self._tn[1:]:
            mps = mps_mult_mpo(mps, mpo, **svdargs)
        mps = mps.reshape(1, -1)
        p = PEPS(mps, self)
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
        output: self._backend.tensor or scalar
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
        for (i, j), tsr in np.ndenumerate(self._tn):
            tn[i,j,0], tn[i,j,1] = tbs.einsvd('abcdpq->abi,icdpq' if (i+j) % 2 == 0 else 'abcdpq->aidpq,bci', tsr)
            tn[i,j,(i+j)%2] = tn[i,j,(i+j)%2].reshape(tn[i,j,(i+j)%2].shape + (1, 1))
        return self._contract_TRG(tn, **svdargs)

    def copy(self):
        """
        Return a deep copy of the PEPS.

        Returns
        -------
        output: PEPS
        """
        tn = np.empty_like(self)
        for idx, tsr in np.ndenumerate(self):
            tn[idx] = tsr.coy()
        return PEPS(tn, self)

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
        tn = np.empty_like(self._tn)
        for idx, tsr in np.ndenumerate(self._tn):
            if not indices or idx in indices:
                tn[idx] = tsr.transpose(0, 1, 2, 3, 5, 4)
            else:
                tn[idx] = tsr.copy()
        return PEPS(tn, self)

    def inner(self, B):
        """
        Compute the inner product of two PEPS, i.e. <psi|phi>. Equivalent to self apply on B^dagger.
        
        Parameters
        ----------
        B: PEPS
            The second PEPS.
        
        Returns
        -------
        output: PEPS
        """
        return self.dagger().apply(B)

    def norm(self):
        """
        Compute the PEPS contracted with its Hermitian conjugate, i.e. <psi|psi>.
        
        Returns
        -------
        output: PEPS
        """
        return self.inner(self)

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
        tn = self._tn
        for _ in range(num_rotate90 % 4):
            tn = np.rot90(tn)
            for idx, tsr in np.ndenumerate(tn):
                tn[idx] = tsr.transpose(1, 2, 3, 0, 4, 5)
        return PEPS(tn, self)

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
    #     tn = self._tn.T
    #     for idx, tsr in np.ndenumerate(tn):
    #         tn[idx] = tsr.transpose(3, 2, 1, 0, 4, 5)
    #     return PEPS(tn, self)

    def _contract_TRG(self, tn, **svdargs):
        # base case
        if tn.shape == (2, 2, 2):
            p = np.empty((2, 2), dtype=object)
            for i, j in np.ndindex((2, 2)):
                p[i,j] = self._backend.einsum('abipq,icdPQ->abcdp+Pq+Q' if (i+j) % 2 == 0 else 'aidpq,bciPQ->abcdp+Pq+Q', tn[i,j][0], tn[i,j][1])
            return PEPS(p, self).contract_BMPS()
        
        # contract specific horizontal and vertical bonds and SVD truncate the generated squared bonds
        for i, j in np.ndindex(tn.shape[:2]):
            if j > 0 and j % 2 == 0:
                k = 1 - i % 2
                l = j - ((i // 2 * 2 + j) % 4 == 0)
                tn[i,l][k] = self._backend.einsum('ibapq,ABiPQ->Ab+Bap+Pq+Q' if k else 'biapq,BAiPQ->b+BAap+Pq+Q', tn[i,j-1][k], tn[i,j][k])
                if i % 2 == 1:
                    tn[i-1,l][1], tn[i,l][0] = self._backend.einsum('aidpq,iBCPQ->aBCdp+Pq+Q', tn[i-1,l][1], tn[i,l][0], **svdargs)
            if i > 0 and i % 2 == 0:
                k = 1 - j % 2
                l = int((i + j // 2 * 2) % 4 == 0)
                tn[i-l,j][l] = self._backend.einsum('biapq,iBAPQ->b+BAap+Pq+Q' if k else 'aibpq,iABPQ->aAb+Bp+Pq+Q', tn[i-1,j][1], tn[i,j][0])
                if j % 2 == 1:
                    tn[i-l,j-1][l], tn[i-l,j][l] = self._backend.einsum('icdpq,ABiPQ->ABcdp+Pq+Q', tn[i-l,j-1][l], tn[i-l,j][l], **svdargs)
        
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
                    new_tn[m,n][1] = self._backend.einsum('abipq,iCAPQ->bCa+Ap+Pq+Q' if i == 0 else 'aibpq,iCBPQ->aCb+Bp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 0 and i % 2 == 1:
                new_tn[m,n][0] = self._backend.einsum('abipq,iBCPQ->ab+BCp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 3 and i % 2 == 0:
                new_tn[m,n][1] = self._backend.einsum('aibpq,ACiPQ->a+ACbp+Pq+Q', tn[i,j][0], tn[i,j][1])
            elif (i + j) % 4 == 3 and i % 2 == 1:
                new_tn[m,n][0] = self._backend.einsum('aibpq,CBiPQ->aCb+Bp+Pq+Q', tn[i,j][0], tn[i,j][1])
            else:
                if new_tn[m,n][0] is None:
                    new_tn[m,n][0] = tn[i,j][0]
                if new_tn[m,n][1] is None:
                    new_tn[m,n][1] = tn[i,j][1]
        
        # SVD truncate the squared bonds generated by the diagonal contractions
        for i, j in np.ndindex(new_tn.shape[:2]):
            if (i + j) % 2 == 0 and new_tn[i,j][0] is not None and new_tn[i,j][1] is not None:
                new_tn[i,j][0], new_tn[i,j][1] = self._backend.einsum('abipq,iCDPQ->abCDp+Pq+Q', new_tn[i,j][0], new_tn[i,j][1], **svdargs)
            elif (i + j) % 2 == 1:
                new_tn[i,j][0], new_tn[i,j][1] = self._backend.einsum('aidpq,BCiPQ->aBCdp+Pq+Q', new_tn[i,j][0], new_tn[i,j][1], **svdargs)

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
        return self._backend.einsum(einstr, a, b, **svdargs)
