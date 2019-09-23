"""
This module implements state vector quantum register.
"""

import numpy as np

from ..quantum_register import QuantumRegister
from ..backends import get_backend
from ..gates import tensorize


class StateVectorQuantumRegister(QuantumRegister):
    def __init__(self, nqubit, backend):
        self.backend = get_backend(backend)
        self.state = self.backend.zeros((2,)*nqubit, dtype=complex)
        self.state[(0,)*nqubit] = 1

    @property
    def nqubit(self):
        return self.backend.ndim(self.state)

    def apply_gate(self, gate):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        input_state_indices = [*range(self.nqubit)]
        def replace_indices(indices):
            for i, q in enumerate(gate.qubits):
                indices[q] = self.nqubit + i
            return indices
        self.backend.einsum(
            self.state, [*range(self.nqubit)],
            tensor, [*gate.qubits, *range(self.nqubit, self.nqubit+len(gate.qubits))],
            replace_indices([*range(self.nqubit)]),
            out=self.state
        )

    def apply_circuit(self, circuit):
        for gate in circuit.gates:
            self.apply_gate(gate)

    def amplitude(self, bits):
        if len(bits) != self.nqubit:
            raise ValueError('bits number and qubits number do not match')
        return self.state[tuple(bits)]

    def probability(self, bits):
        return np.abs(self.amplitude(bits))**2

    def probabilities(self):
        prob_vector = np.real(self.state)**2 + np.imag(self.state)**2
        return [(index, a) for index, a in np.ndenumerate(state) if not np.isclose(a.conj()*a,0)]
