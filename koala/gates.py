"""
This module defines the gates supported by the simulators and their tensor
representations.
"""

from functools import lru_cache

from . import tensors


def tensorize(backend, gate_name, *gate_parameters):
    if gate_name not in _GATES:
        raise ValueError(f"{gate_name} gate is not supported")
    return _GATES[gate_name](backend, *gate_parameters)


_GATES = {}

def _register(func):
    _GATES[func.__name__] = func
    return func

@_register
@lru_cache(maxsize=None)
def H(backend):
    return backend.astensor(tensors.H())

@_register
@lru_cache(maxsize=None)
def X(backend):
    return backend.astensor(tensors.X())

@_register
@lru_cache(maxsize=None)
def Y(backend):
    return backend.astensor(tensors.Y())

@_register
@lru_cache(maxsize=None)
def Z(backend):
    return backend.astensor(tensors.Z())

@_register
@lru_cache(maxsize=None)
def S(backend):
    return backend.astensor(tensors.S())

@_register
@lru_cache(maxsize=None)
def Sdag(backend):
    return backend.astensor(tensors.Sdag())

@_register
@lru_cache(maxsize=None)
def T(backend):
    return backend.astensor(tensors.T())

@_register
@lru_cache(maxsize=None)
def Tdag(backend):
    return backend.astensor(tensors.Tdag())

@_register
@lru_cache(maxsize=None)
def W(backend):
    return backend.astensor(tensors.W())

@_register
@lru_cache(maxsize=None)
def sqrtX(backend):
    return backend.astensor(tensors.sqrtX())

@_register
@lru_cache(maxsize=None)
def sqrtY(backend):
    return backend.astensor(tensors.sqrtY())

@_register
@lru_cache(maxsize=None)
def sqrtZ(backend):
    return backend.astensor(tensors.sqrtZ())

@_register
@lru_cache(maxsize=None)
def sqrtW(backend):
    return backend.astensor(tensors.sqrtW())

@_register
@lru_cache(maxsize=64)
def R(backend, theta):
    return backend.astensor(tensors.R(theta))

@_register
@lru_cache(maxsize=64)
def U1(backend, lmbda):
    return backend.astensor(tensors.U1(lmbda))

@_register
@lru_cache(maxsize=64)
def U2(backend, phi, lmbda):
    return backend.astensor(tensors.U2(phi, lmbda))

@_register
@lru_cache(maxsize=64)
def U3(backend, theta, phi, lmbda):
    return backend.astensor(tensors.U3(theta, phi, lmbda))

@_register
@lru_cache(maxsize=None)
def CH(backend):
    return backend.astensor(tensors.control(1, tensors.H()))

@_register
@lru_cache(maxsize=None)
def CX(backend):
    return backend.astensor(tensors.control(1, tensors.X()))

@_register
@lru_cache(maxsize=None)
def CY(backend):
    return backend.astensor(tensors.control(1, tensors.Y()))

@_register
@lru_cache(maxsize=None)
def CZ(backend):
    return backend.astensor(tensors.control(1, tensors.Z()))

@_register
@lru_cache(maxsize=64)
def CR(backend, theta):
    return backend.astensor(tensors.control(1, tensors.R(theta)))

@_register
@lru_cache(maxsize=64)
def CU1(backend, lmbda):
    return backend.astensor(tensors.control(1, tensors.U1(lmbda)))

@_register
@lru_cache(maxsize=64)
def CU2(backend, phi, lmbda):
    return backend.astensor(tensors.control(1, tensors.U2(phi, lmbda)))

@_register
@lru_cache(maxsize=64)
def CU3(backend, theta, phi, lmbda):
    return backend.astensor(tensors.control(1, tensors.U3(theta, phi, lmbda)))

@_register
@lru_cache(maxsize=None)
def SWAP(backend):
    return backend.astensor(tensors.SWAP())

@_register
@lru_cache(maxsize=None)
def ISWAP(backend):
    return backend.astensor(tensors.ISWAP())
