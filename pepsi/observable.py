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
    def X(position):
        return Observable([(tensors.X(), (position,))])

    @staticmethod
    def Y(position):
        return Observable([(tensors.Y(), (position,))])

    @staticmethod
    def Z(position):
        return Observable([(tensors.Z(), (position,))])

    @staticmethod
    def operator(tensor, positions):
        if tensor.ndim != len(positions) * 2:
            raise ValueError(f'tensor shape and number of positions do not match')
        return Observable([(tensor, positions)])

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
            f'{operator},{positions}'
            for operator, positions in self.operators
        )
        return f"Observable({operators_str})"
