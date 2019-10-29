"""
This module defines NumPy backend.
"""

import numpy as np
import numpy.linalg as la

from .backend import Backend


class NumPyBackend(Backend):
    def shape(self, a):
        return a.shape

    def ndim(self, a):
        return a.ndim

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype):
        return np.ones(shape, dype=dtype)

    def astensor(self, a):
        return np.asarray(a)

    def copy(self, a):
        return np.copy(a)

    def reshape(self, newshape):
        return np.reshape(a, newshape)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes)

    def conjugate(self, a):
        return np.conjugate(a)

    def einsum(self, *args, **kwargs):
        return np.einsum(*args, **kwargs)

    def einsvd(self, einstr, A, rank=None, threshold=None, size_limit=None, criterion=None, mult_s=True, rescale=False):
        """
        Perform Singular Value Decomposition according to the specified Einstein notation string. 
        Will always preserve at least one singular value during the truncation.

        Parameters
        ----------
        einstr: str
            A string of Einstein notations in the form of 'idxofA->idxofU,idxofV'. There must be one and only one contraction index.
        A: tensor_like
            The tensor to be decomposed. Should be of order 2 or more.
        rank: int or None, optional
            The minimum number of singular values/vectors to preserve. Will influence the actual truncation rank.
        threshold: float or None, optional
            The value used with criterion to decide the cutoff. Will influence the actual truncation rank.
        size_limit: int or tuple or None, optional
            The size limit(s) for both U and V (when specified as a int) or U and V respectively (when specified as a tuple).
            Will influence the actual truncation rank.
        criterion: int or None, optional
            The norm to be used together with threshold to decide the cutoff. Will influence the actual truncation rank.
            When being left as None, the threshold is treated as the plain cutoff value.
            Otherwise, cutoff rank is the largest int satisfies: threshold * norm(s) > norm(s[rank:]).
        mult_s: bool, optional
            Whether or not to multiply U and V by S**0.5 to decompose A into two tensors instead of three. True by default.

        Returns
        -------
        u: tensor_like
            A unitary tensor with indices ordered by the Einstein notation string.
        s: 1d tensor_like
            A 1d tensor containing singular values sorted in descending order.
        v: tensor_like
            A unitary tensor with indices ordered by the Einstein notation string.
        """
        str_a, str_uv = einstr.replace(' ', '').split('->')
        str_u, str_v = str_uv.split(',')
        char_i = list(set(str_v) - set(str_a))[0]

        # transpose and reshape A into u.size * v.size and do svd
        u, s, vh = la.svd(np.einsum(str_a + '->' + (str_u + str_v).replace(char_i, ''), A).reshape((-1, np.prod([A.shape[str_a.find(c)] for c in str_v if c != char_i], dtype=int))), full_matrices=False)
        rank = rank or len(s)

        if size_limit is not None:
            if np.isscalar(size_limit):
                size_limit = (size_limit, size_limit)
            if size_limit[0] is not None:
                rank = min(rank, int(size_limit[0] / u.shape[0]) or 1)
            if size_limit[1] is not None:
                rank = min(rank, int(size_limit[1] / vh.shape[1]) or 1)

        if threshold is not None:
            # will always preserve at least one singular value
            if criterion is None:
                rank = 1 if threshold > s[0] else min((rank, len(s) - np.searchsorted(s[::-1], threshold)))
            else:
                threshold = threshold * la.norm(s, criterion)
                for i in range(rank, 0, -1):
                    if la.norm(s[i-1:], criterion) >= threshold:
                        rank = i
                        break

        if rank < len(s):
            u = u[:,:rank]
            s = s[:rank] * (la.norm(s) / la.norm(s[:rank])) if rescale else s[:rank]
            vh = vh[:rank]

        if mult_s:
            sqrtS = np.diag(s ** 0.5)
            u = np.dot(u, sqrtS)
            vh = np.dot(sqrtS, vh)

        # reshape and transpose u and vh into tgta and tgtb
        u = np.einsum(str_u.replace(char_i, '') + char_i + '->' + str_u, u.reshape([A.shape[str_a.find(c)] for c in str_u if c != char_i] + [-1]))
        vh = np.einsum(char_i + str_v.replace(char_i, '') + '->' + str_v, vh.reshape([-1] + [A.shape[str_a.find(c)] for c in str_v if c != char_i]))
        return u, s, vh

    def norm2(self, a):
        return la.norm(a)

    # def einsvd(self, A, inds, **kwargs):
    #     """Do svd on tensor A with indices specified indices in u and others in sv.
    #     Indices in u are in the order provided in `inds`.
    #     Indices in sv are in the same ordering as in original tensor A. 

    #     # Arguments:
    #     A - multi dimensional array
    #     inds - list of indices that will go to u after svd

    #     # Example:
    #     >>> A = np.random.rand(1,2,3,4,5,6,7)
    #     >>> u, sv = einsvd(A, [2,4,6])
    #     >>> u.shape
    #     (3, 5, 7, 48)
    #     >>> sv.shape
    #     (48, 1, 2, 4, 6)
    #     """
    #     B = np.moveaxis(A, inds, [*range(len(inds))])
    #     left_dim = np.prod(B.shape[:len(inds)])
    #     shape = B.shape
    #     B = np.reshape(B, (left_dim, -1))
    #     u,s,v = np.linalg.svd(B, full_matrices=False)
    #     s, _ = self.truncate(s, **kwargs)
    #     u = np.reshape(u[:,:len(s)], (*shape[:len(inds)], -1))
    #     sv = np.reshape(np.diag(s)@v[:len(s), :], (-1, *shape[len(inds):]))
    #     return u, sv

    # def truncate(self, s, **kwargs): #cutoff = 0.0, mode = "abs", maxdim = -1, mindim = 1
    #     """Truncate array `s` according to given rule. maxdim/ mindim has
    #     higher priority than absolute/relative cutoff.

    #     # Parameters: 
    #     - `s`: array (of singular values)
    #     - `cutoff`: singular values smaller than this value will be truncated
    #     (actual operation depend on mode)
    #     - `mode`: define truncation mode. 
    #         "abs" : throw away all singular values < `cutoff`
    #         "rel" : `cutoff` in [0,1] is interpreted as a percentage
    #         (e.g. cutoff=0.5 in "rel" mode means throw away smal singular values
    #          s.t. norm(remaining vector) = 0.5*norm(original vector))
    #     - `maxdim`: regardless of other rules, number of sigular values are
    #     capped at `maxdim`
    #     - `mindim`: regardless of other rules, will keep at least `mindim`
    #     singular values
    #     """
    #     cutoff = kwargs.get('cutoff', 0.0)
    #     mode = kwargs.get('mode', "abs")
    #     maxdim = kwargs.get('maxdim', len(s))
    #     mindim = kwargs.get('mindim', 1)

    #     p = np.argsort(s)[::-1]
    #     s = s[p]
    #     s = np.clip(s, 0.0, None) # zero out negative singular values for safety
    #     s_trunc = s
    #     if mode == 'abs' and cutoff >= 0.0:
    #         s_trunc = s[s>=cutoff]
    #     elif mode == 'rel':
    #         cutoff_val = sum(s**2)*(1-cutoff)
    #         cut = 0.0
    #         ind = len(s)-1
    #         while True:
    #             cut += s[ind]**2
    #             if cut > cutoff_val:
    #                 break
    #             else:
    #                 ind-=1
    #         s_trunc = np.resize(s, (ind+2))
    #     else:
    #         raise ValueError("Unknown truncation mode")
    #     if len(s_trunc) < mindim:
    #         s_trunc = s[:mindim+1]
    #     if len(s_trunc) > maxdim:
    #         s_trunc = s[:maxdim]
    #     truncated_n = len(s) - len(s_trunc)
    #     return s_trunc, truncated_n
