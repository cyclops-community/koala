"""
This module implements state vector quantum register.
"""

from string import ascii_letters as chars
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
        self.state = apply_operator(self.backend, self.state, tensor, gate.qubits)

    def apply_circuit(self, circuit):
        for gate in circuit.gates:
            self.apply_gate(gate)

    def apply_operator(self, operator, qubits):
        self.state = apply_operator(self.backend, self.state, operator, qubits)

    def normalize(self):
        self.state /= self.backend.norm2(self.state)

    def amplitude(self, bits):
        if len(bits) != self.nqubit:
            raise ValueError('bits number and qubits number do not match')
        return self.state[tuple(bits)]

    def probability(self, bits):
        return np.abs(self.amplitude(bits))**2

    def expectation(self, observable):
        e = 0
        all_terms = ''.join(chars[i] for i in range(self.nqubit))
        einstr = f'{all_terms},{all_terms}'
        for tensor, qubits in observable:
            state = self.backend.copy(self.state)
            state = apply_operator(self.backend, state, self.backend.astensor(tensor), qubits)
            e += np.real_if_close(self.backend.einsum(
                einstr, state, self.backend.conjugate(self.state), 
            ))
        return e

    def probabilities(self):
        prob_vector = np.real(self.state)**2 + np.imag(self.state)**2
        return [(index, a) for index, a in np.ndenumerate(state) if not np.isclose(a.conj()*a,0)]


def apply_operator(backend, state, operator, axes):
    ndim = backend.ndim(state)
    input_state_indices = range(ndim)
    operator_indices = [*axes, *range(ndim, ndim+len(axes))]
    output_state_indices = [*range(ndim)]
    for i, axis in enumerate(axes):
        output_state_indices[axis] = i + ndim
    input_terms = ''.join(chars[i] for i in input_state_indices)
    operator_terms = ''.join(chars[i] for i in operator_indices)
    output_terms = ''.join(chars[i] for i in output_state_indices)
    einstr = f'{input_terms},{operator_terms}->{output_terms}'
    return backend.einsum(einstr, state, operator)
