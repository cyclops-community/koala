"""
This module defines the gates supported by the simulators and their tensor
representations.
"""

from functools import lru_cache


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
    return backend.astensor(_H())

@_register
@lru_cache(maxsize=None)
def X(backend):
    return backend.astensor(_X())

@_register
@lru_cache(maxsize=None)
def Y(backend):
    return backend.astensor(_Y())

@_register
@lru_cache(maxsize=None)
def Z(backend):
    return backend.astensor(_Z())

@_register
@lru_cache(maxsize=64)
def R(backend, theta):
    return backend.astensor(_R(theta))

@_register
@lru_cache(maxsize=None)
def CH(backend):
    return backend.astensor(_CH())

@_register
@lru_cache(maxsize=None)
def CY(backend):
    return backend.astensor(_CY())

@_register
@lru_cache(maxsize=None)
def CZ(backend):
    return backend.astensor(_CZ())

@_register
@lru_cache(maxsize=64)
def CR(backend, theta):
    return backend.astensor(_CR(theta))

@_register
@lru_cache(maxsize=None)
def SWAP(backend):
    return backend.astensor(_SWAP())


# =============================================================================
# Raw gate tensor constructors
# -----------------------------------------------------------------------------
from itertools import repeat
import numpy as np

def _add_control(nctrl, tensor):
    nqubit = nctrl + tensor.ndim // 2
    result = np.eye(2**nqubit, dtype=complex)
    result[2**nctrl:, 2**nctrl:] = tensor
    return result.reshape(*repeat(2, nqubit*2))

def _H():
    return (np.array([1,1,1,-1],dtype=complex)/np.sqrt(2)).reshape(2,2)

def _X():
    return np.array([0,1,1,0],dtype=complex).reshape(2,2)

def _Y():
    return np.array([0,-1j,1j,0],dtype=complex).reshape(2,2)

def _Z():
    return np.array([1,0,0,-1],dtype=complex).reshape(2,2)

def _R(theta):
    return np.array([1,0,0,np.exp(1j*theta)],dtype=complex).reshape(2,2)

def _CH():
    return _add_control(1, _H())

def _CX():
    return _add_control(1, _X())

def _CY():
    return _add_control(1, _Y())

def _CZ():
    return _add_control(1, _Z())

def _CR(theta):
    return _add_control(1, _R(theta))

def _SWAP():
    return np.array([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],dtype=complex).reshape(2,2,2,2)
# =============================================================================

