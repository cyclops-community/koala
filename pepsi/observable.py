"""
"""


import numpy as np

from . import tensors


class Observable:
    def __init__(self, operators):
        self.operators = operators

    @staticmethod
    def identity():
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
        result = Observable.identity()
        for observable in observables:
            result += observable
        return result

    def __iter__(self):
        yield from self.operators

    def __add__(self, other):
        return Observable([*self, *other])

    def __iadd__(self, other):
        self.operators.extend(other.operators)
        return self

    def __str__(self):
        operators_str = ';'.join(
            f'{operator},{qubits}'
            for operator, qubits in self.operators
        )
        return f"Observable({operators_str})"
