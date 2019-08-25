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
    return TensorGate(
        tensor=np.eye(2),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: H):
    return TensorGate(
        tensor=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: X):
    return TensorGate(
        tensor=np.array([[0,1],[1,0]],dtype=complex),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: Y):
    return TensorGate(
        tensor=np.array([[0,-1j],[1j,0]],dtype=complex),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: Z):
    return TensorGate(
        tensor=np.array([[1,0],[0,-1]],dtype=complex),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: S):
    return TensorGate(
        tensor=np.array([[1,0],[0,1j]],dtype=complex),
        qubits=(gate.qubit,)
    )


@tensorize.register
def _(gate: CX):
    return TensorGate(
        tensor=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex),
        qubits=(gate.ctrl, gate.target)
    )

