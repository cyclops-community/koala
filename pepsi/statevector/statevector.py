"""
This module implements state vector quantum register.
"""

from numbers import Number
from string import ascii_letters as chars
import numpy as np

import tensorbackends

from ..quantum_state import QuantumState
from ..gates import tensorize


class StateVector(QuantumState):
    def __init__(self, tensor, backend):
        self.backend = tensorbackends.get(backend)
        self.tensor = tensor

    @property
    def nsite(self):
        return self.tensor.ndim
    
    def copy(self):
        return StateVector(self.tensor.copy(), self.backend)

    def conjugate(self):
        return StateVector(selkf.tensor.conj(), self.backend)

    def apply_gate(self, gate):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        self.apply_operator(tensor, gate.qubits)

    def apply_circuit(self, gates):
        for gate in gates:
            self.apply_gate(gate)

    def apply_operator(self, operator, sites):
        self.tensor = apply_operator(self.backend, self.tensor, operator, sites)

    def norm(self):
        return self.backend.norm(self.tensor)

    def __imul__(self, a):
        if isinstance(a, Number):
            self.tensor *= a
            return self
        else:
            return NotImplemented

    def __itruediv__(self, a):
        if isinstance(a, Number):
            self.tensor /= a
            return self
        else:
            return NotImplemented

    def amplitude(self, indices):
        if len(indices) != self.nsite:
            raise ValueError('indices number and sites number do not match')
        return self.tensor[tuple(indices)]

    def probability(self, indices):
        return np.abs(self.amplitude(indices))**2

    def expectation(self, observable):
        e = 0
        all_terms = ''.join(chars[i] for i in range(self.nsite))
        einstr = f'{all_terms},{all_terms}->'
        for operator, sites in observable:
            tensor = self.tensor.copy()
            tensor = apply_operator(self.backend, tensor, operator, sites)
            e += np.real_if_close(self.backend.einsum(einstr, tensor, self.tensor.conj()))
        return e

    def probabilities(self):
        prob_vector = np.real(self.tensor)**2 + np.imag(self.tensor)**2
        return [(index, a) for index, a in np.ndenumerate(self.tensor) if not np.isclose(a.conj()*a,0)]


def apply_operator(backend, state_tensor, operator, axes):
    operator = backend.astensor(operator)
    ndim = state_tensor.ndim
    input_state_indices = range(ndim)
    operator_indices = [*axes, *range(ndim, ndim+len(axes))]
    output_state_indices = [*range(ndim)]
    for i, axis in enumerate(axes):
        output_state_indices[axis] = i + ndim
    input_terms = ''.join(chars[i] for i in input_state_indices)
    operator_terms = ''.join(chars[i] for i in operator_indices)
    output_terms = ''.join(chars[i] for i in output_state_indices)
    einstr = f'{input_terms},{operator_terms}->{output_terms}'
    return backend.einsum(einstr, state_tensor, operator)
