"""
This module defines CTF backend.
"""

import string

import ctf
import numpy as np

from .backend import Backend


class CTFBackend(Backend):
    def shape(self, a):
        return a.shape

    def ndim(self, a):
        return a.ndim

    def empty(self, shape, dtype):
        return ctf.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return ctf.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype):
        return ctf.ones(shape, dype=dtype)

    def astensor(self, a):
        return ctf.astensor(a)

    def copy(self, a):
        return ctf.copy(a)

    def reshape(self, newshape):
        return ctf.reshape(a, newshape)

    def transpose(self, a, axes=None):
        return ctf.transpose(a, axes)

    def conjugate(self, a):
        return ctf.conj(a)

    def einsum(self, *args, **kwargs):
        return ctf.einsum(*args, **kwargs)

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
        shape_u = np.prod([A.shape[str_a.find(c)] for c in str_u if c != char_i])
        shape_v = np.prod([A.shape[str_a.find(c)] for c in str_v if c != char_i])

        rank = rank or min(shape_u, shape_v)

        if size_limit is not None:
            if np.isscalar(size_limit):
                size_limit = (size_limit, size_limit)
            if size_limit[0] is not None:
                rank = min(rank, int(size_limit[0] / shape_u) or 1)
            if size_limit[1] is not None:
                rank = min(rank, int(size_limit[1] / shape_v) or 1)

        if threshold is None or criterion is None:
            u, s, vh = A.i(str_a).svd(str_u, str_v, rank, threshold)
        else:
            u, s, vh = A.i(str_a).svd(str_u, str_v)
            threshold = threshold * ctf.norm(s, criterion)
            # will always preserve at least one singular value
            for i in range(rank, 0, -1):
                if ctf.norm(s[i-1:], criterion) >= threshold:
                    rank = i
                    break;
            if rank < s.size:
                u = u[tuple(slice(None) for i in range(str_u.find(char_i))) + (slice(0, rank),)]
                s = s[:rank] * (s.norm2() / s[:rank].norm2()) if rescale else s[:rank]
                vh = vh[tuple(slice(None) for i in range(str_v.find(char_i))) + (slice(0, rank),)]

        if mult_s:
            char_s = list(set(string.ascii_letters) - set(str_v))[0]
            sqrtS = ctf.diag(s ** 0.5)
            vh = ctf.einsum(char_s + char_i + ',' + str_v + '->' + str_v.replace(char_i, char_s), sqrtS, vh)
            char_s = list(set(string.ascii_letters) - set(str_u))[0]
            u = ctf.einsum(str_u + ',' + char_s + char_i + '->' + str_u.replace(char_i, char_s), u, sqrtS)

        return u, s, vh

    def norm2(self, a):
        return a.norm2()
