"""
This module defines the interface of a backend.
"""

class Backend:
    def shape(self, a):
        raise NotImplementedError()

    def ndim(self, a):
        raise NotImplementedError()

    def empty(self, shape, dtype):
        raise NotImplementedError()

    def zeros(self, shape, dtype):
        raise NotImplementedError()

    def ones(self, shape, dtype):
        raise NotImplementedError()

    def astensor(self, a):
        raise NotImplementedError()

    def copy(self, a):
        raise NotImplementedError()

    def reshape(self, newshape):
        raise NotImplementedError()

    def transpose(self, a, axes=None):
        raise NotImplementedError()

    def conjugate(self, a):
        raise NotImplementedError()

    def einsum(self, *args, **kwargs):
        raise NotImplementedError()

    def einsvd(self):
        raise NotImplementedError()

    def norm2(self):
        raise NotImplementedError()
