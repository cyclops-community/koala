"""
"""

from numbers import Real

import numpy as np

from . import tensors


class Observable:
    def __init__(self, operators):
        self.operators = operators

    @staticmethod
    def zero():
        return Observable([])

    @staticmethod
    def X(qubit):
        return Observable([(tensors.X(), (qubit,))])

    @staticmethod
    def Y(qubit):
        return Observable([(tensors.Y(), (qubit,))])

    @staticmethod
    def Z(qubit):
        return Observable([(tensors.Z(), (qubit,))])

    @staticmethod
    def XX(first, second):
        return Observable([(tensors.XX(), (first, second))])

    @staticmethod
    def XY(first, second):
        return Observable([(tensors.XY(), (first, second))])

    @staticmethod
    def XZ(first, second):
        return Observable([(tensors.XZ(), (first, second))])

    @staticmethod
    def YY(first, second):
        return Observable([(tensors.YY(), (first, second))])

    @staticmethod
    def YZ(first, second):
        return Observable([(tensors.YZ(), (first, second))])

    @staticmethod
    def ZZ(first, second):
        return Observable([(tensors.ZZ(), (first, second))])

    @staticmethod
    def operator(tensor, qubits):
        if tensor.ndim != len(qubits) * 2:
            raise ValueError(f'tensor shape and number of target qubits do not match')
        return Observable([(tensor, qubits)])

    @staticmethod
    def sum(observables):
        result = Observable.zero()
        for observable in observables:
            result += observable
        return result

    def __iter__(self):
        yield from self.operators

    def scale(self, a):
        return Observable([(tensor*a, qubits) for tensor, qubits in self.operators])

    def __pos__(self):
        return Observable([*self.operators])

    def __neg__(self):
        return Observable([(-tensor, qubits) for tensor, qubits in self.operators])

    def __add__(self, other):
        return Observable([*self, *other])

    def __iadd__(self, other):
        self.operators.extend(other.operators)
        return self

    def __mul__(self, other):
        if isinstance(other, Real):
            return self.scale(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            return self.scale(other)
        return NotImplemented

    def __str__(self):
        operators_str = ';'.join(
            f'{operator},{qubits}'
            for operator, qubits in self.operators
        )
        return f"Observable({operators_str})"
