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
@lru_cache(maxsize=None)
def S(backend):
    return backend.astensor(_S())

@_register
@lru_cache(maxsize=None)
def Sdag(backend):
    return backend.astensor(_Sdag())

@_register
@lru_cache(maxsize=None)
def T(backend):
    return backend.astensor(_T())

@_register
@lru_cache(maxsize=None)
def Tdag(backend):
    return backend.astensor(_Tdag())

@_register
@lru_cache(maxsize=64)
def R(backend, theta):
    return backend.astensor(_R(theta))

@_register
@lru_cache(maxsize=64)
def U1(backend, lmbda):
    return backend.astensor(_U1(lmbda))

@_register
@lru_cache(maxsize=64)
def U2(backend, phi, lmbda):
    return backend.astensor(_U2(phi, lmbda))

@_register
@lru_cache(maxsize=64)
def U3(backend, theta, phi, lmbda):
    return backend.astensor(_U3(theta, phi, lmbda))

@_register
@lru_cache(maxsize=None)
def CH(backend):
    return backend.astensor(_control(1, _H()))

@_register
@lru_cache(maxsize=None)
def CX(backend):
    return backend.astensor(_control(1, _X()))

@_register
@lru_cache(maxsize=None)
def CY(backend):
    return backend.astensor(_control(1, _Y()))

@_register
@lru_cache(maxsize=None)
def CZ(backend):
    return backend.astensor(_control(1, _Z()))

@_register
@lru_cache(maxsize=64)
def CR(backend, theta):
    return backend.astensor(_control(1, _R(theta)))

@_register
@lru_cache(maxsize=64)
def CU1(backend, lmbda):
    return backend.astensor(_control(1, _U1(lmbda)))

@_register
@lru_cache(maxsize=64)
def CU2(backend, phi, lmbda):
    return backend.astensor(_control(1, _U2(phi, lmbda)))

@_register
@lru_cache(maxsize=64)
def CU3(backend, theta, phi, lmbda):
    return backend.astensor(_control(1, _U3(theta, phi, lmbda)))

@_register
@lru_cache(maxsize=None)
def SWAP(backend):
    return backend.astensor(_SWAP())


# =============================================================================
# Raw gate tensor constructors
# -----------------------------------------------------------------------------
from itertools import repeat
import numpy as np

def _control(nctrl, tensor):
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

def _S():
    return np.array([1,0,0,1j],dtype=complex).reshape(2,2)

def _Sdag():
    return np.array([1,0,0,-1j],dtype=complex).reshape(2,2)

def _T():
    return _U1(np.pi/4)

def _Tdag():
    return _U1(-np.pi/4)

def _R(theta):
    return _U1(theta)

def _U1(lmbda):
    return _U3(0, 0, lmbda)

def _U2(phi, lmbda):
    return _U3(np.pi/2, phi, lmbda)

def _U3(theta, phi, lmbda):
    c, s = np.cos(theta), np.sin(theta)
    e_phi, e_lmbda = np.exp(1j*phi), np.exp(1j*lmbda)
    return np.array([c,-e_lmbda*s,e_phi*s,e_lmbda*e_phi*c],dtype=complex).reshape(2,2)

def _SWAP():
    return np.array([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],dtype=complex).reshape(2,2,2,2)
# =============================================================================
