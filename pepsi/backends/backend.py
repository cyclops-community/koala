"""
This module defines the interface of a backend.
"""

class Backend:
    @staticmethod
    def shape(a):
        raise NotImplementedError()

    @staticmethod
    def ndim(a):
        raise NotImplementedError()

    @staticmethod
    def empty(shape, dtype):
        raise NotImplementedError()

    @staticmethod
    def zeros(shape, dtype):
        raise NotImplementedError()

    @staticmethod
    def ones(shape, dtype):
        raise NotImplementedError()

    @staticmethod
    def astensor(a):
        raise NotImplementedError()

    @staticmethod
    def einsum(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def einsvd():
        raise NotImplementedError()
