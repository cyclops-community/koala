"""
This module defines NumPy backend.
"""

import numpy as np

from .backend import Backend


class NumPyBackend(Backend):
    @staticmethod
    def shape(a):
        return a.shape

    @staticmethod
    def ndim(a):
        return a.ndim

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype):
        return np.ones(shape, dype=dtype)

    @staticmethod
    def astensor(A):
        return np.asarray(A)

    @staticmethod
    def einsum(*args, **kwargs):
        return np.einsum(*args, **kwargs)
