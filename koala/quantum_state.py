"""
This module defines the interface of a quantum state.
"""

class QuantumState:
    @property
    def nsite(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def norm(self):
        raise NotImplementedError()

    def conjugate(self):
        raise NotImplementedError()

    def apply_gate(self, gate):
        raise NotImplementedError()

    def apply_circuit(self, gates):
        raise NotImplementedError()

    def apply_operator(self, operator, sites):
        raise NotImplementedError()

    def __imul__(self, a):
        raise NotImplementedError()

    def __itruediv__(self, a):
        raise NotImplementedError()

    def amplitude(self, indices):
        raise NotImplementedError()

    def probability(self, indices):
        raise NotImplementedError()

    def expectation(self, observable):
        raise NotImplementedError()
