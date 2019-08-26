import functools
from collections import namedtuple

import numpy as np

from circuit import I, H, X, Y, Z, S, T, R, Rx, Ry, Rz
from circuit import CX, CY, CZ, SWAP
from circuit import CCX


TensorGate = namedtuple('TensorGate', ['tensor', 'qubits'])


@functools.singledispatch
def tensorize(gate):
    raise TypeError(f'Unknown gate {type(gate).__name__}')


@tensorize.register
def _(gate: I):
    return TensorGate(I_tensor, (gate.qubit,))


@tensorize.register
def _(gate: H):
    return TensorGate(H_tensor, (gate.qubit,))


@tensorize.register
def _(gate: X):
    return TensorGate(X_tensor, (gate.qubit,))


@tensorize.register
def _(gate: Y):
    return TensorGate(Y_tensor, (gate.qubit,))


@tensorize.register
def _(gate: Z):
    return TensorGate(Z_tensor, (gate.qubit,))


@tensorize.register
def _(gate: S):
    return TensorGate(S_tensor, (gate.qubit,))


@tensorize.register
def _(gate: CX):
    return TensorGate(CX_tensor, (gate.ctrl, gate.target))


I_tensor = np.eye(2,dtype=complex)
H_tensor = np.array([1,1,1,-1],dtype=complex)/np.sqrt(2).reshape(2,2)
X_tensor = np.array([0,1,1,0],dtype=complex).reshape(2,2)
Y_tensor = np.array([0,-1j,1j,0],dtype=complex).reshape(2,2)
Z_tensor = np.array([1,0,0,-1],dtype=complex).reshape(2,2)
S_tensor = np.array([1,0,0,1j],dtype=complex).reshape(2,2)
CX_tensor = np.array([1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0],dtype=complex).reshape(2,2,2,2)

