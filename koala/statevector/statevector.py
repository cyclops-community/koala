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

    def amplitude(self, indices):
        if len(indices) != self.nsite:
            raise ValueError('indices number and sites number do not match')
        return self.tensor[tuple(indices)]

    def probability(self, indices):
        return abs(self.amplitude(indices))**2

    def expectation(self, observable):
        return braket(self, observable, self).real

    def probabilities(self):
        prob_vector = np.real(self.tensor)**2 + np.imag(self.tensor)**2
        return [(index, a) for index, a in np.ndenumerate(self.tensor) if not np.isclose(a.conj()*a,0)]

    def inner(self, other):
        terms = ''.join(chars[i] for i in range(self.nsite))
        subscripts = terms + ',' + terms + '->'
        return self.backend.einsum(subscripts, self.tensor.conj(), other.tensor)


def apply_operator(backend, state_tensor, operator, axes):
    operator = backend.astensor(operator)
    ndim = state_tensor.ndim
    input_state_indices = range(ndim)
    operator_indices = [*range(ndim, ndim+len(axes)), *axes]
    output_state_indices = [*range(ndim)]
    for i, axis in enumerate(axes):
        output_state_indices[axis] = i + ndim
    input_terms = ''.join(chars[i] for i in input_state_indices)
    operator_terms = ''.join(chars[i] for i in operator_indices)
    output_terms = ''.join(chars[i] for i in output_state_indices)
    einstr = f'{input_terms},{operator_terms}->{output_terms}'
    return backend.einsum(einstr, state_tensor, operator)


def braket(p, observable, q):
    if p.backend != q.backend:
        raise ValueError('two states must use the same backend')
    if p.nsite != q.nsite:
        raise ValueError('number of sites must be equal in both states')
    all_terms = ''.join(chars[i] for i in range(q.nsite))
    einstr = f'{all_terms},{all_terms}->'
    p_tensor_conj = p.tensor.conj()
    e = 0
    for operator, sites in observable:
        r = apply_operator(q.backend, q.tensor, operator, sites)
        e += p.backend.einsum(einstr, p_tensor_conj, r)
    return e


def inherit_unary_operators(*operator_names):
    def add_unary_operator(operator_name):
        def method(self):
            return StateVector(getattr(self.tensor, operator_name)(), self.backend)
        method.__module__ = StateVector.__module__
        method.__qualname__ = '{}.{}'.format(StateVector.__qualname__, operator_name)
        method.__name__ = operator_name
        setattr(StateVector, operator_name, method)
    for op_name in operator_names:
        add_unary_operator(op_name)


def inherit_binary_operators(*operator_names):
    def add_binary_operator(operator_name):
        def method(self, other):
            if isinstance(other, StateVector) and self.backend == other.backend:
                return StateVector(getattr(self.tensor, operator_name)(other.tensor), self.backend)
            elif isinstance(other, Number):
                return StateVector(getattr(self.tensor, operator_name)(other), self.backend)
            else:
                return NotImplemented
        method.__module__ = StateVector.__module__
        method.__qualname__ = '{}.{}'.format(StateVector.__qualname__, operator_name)
        method.__name__ = operator_name
        setattr(StateVector, operator_name, method)
    for op_name in operator_names:
        add_binary_operator(op_name)


inherit_unary_operators(
    '__pos__',
    '__neg__',
)

inherit_binary_operators(
    '__add__',
    '__sub__',
    '__mul__',
    '__truediv__',
    '__pow__',

    '__radd__',
    '__rsub__',
    '__rmul__',
    '__rtruediv__',
    '__rpow__',

    '__iadd__',
    '__isub__',
    '__imul__',
    '__itruediv__',
    '__ipow__',
)
