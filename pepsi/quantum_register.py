"""
This module defines the interface of a quantum register.
"""

class QuantumRegister:
    def apply_gate(self, gate):
        raise NotImplementedError()

    def apply_circuit(self, circuit):
        raise NotImplementedError()
