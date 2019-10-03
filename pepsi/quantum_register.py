"""
This module defines the interface of a quantum register.
"""

class QuantumRegister:
    @property
    def nqubit(self):
        raise NotImplementedError()

    def apply_gate(self, gate):
        raise NotImplementedError()

    def apply_circuit(self, circuit):
        raise NotImplementedError()

    def apply_operator(self, operator, qubits):
        raise NotImplementedError()

    def measure(self, qubits):
        raise NotImplementedError()

    def sample(self, qubits, k):
        raise NotImplementedError()

    def amplitude(self, bits):
        raise NotImplementedError()

    def probability(self, bits):
        raise NotImplementedError()

    def expectation(self, observable):
        raise NotImplementedError()
