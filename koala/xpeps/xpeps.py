"""
This module defines XPEPS and operations on it.
"""

import random
from math import sqrt
from numbers import Number
from string import ascii_letters as chars

import numpy as np
import tensorbackends

from ..quantum_state import QuantumState
from ..gates import tensorize
from . import contract, update


class XPEPS(QuantumState):
    def __init__(self, sites, horizontal_bonds, vertical_bonds, backend):
        self.backend = tensorbackends.get(backend)
        self.sites = sites
        self.horizontal_bonds = horizontal_bonds
        self.vertical_bonds = vertical_bonds

    @property
    def nrow(self):
        return self.sites.shape[0]

    @property
    def ncol(self):
        return self.sites.shape[1]

    @property
    def shape(self):
        return self.sites.shape

    @property
    def nsite(self):
        return self.nrow * self.ncol

    def copy(self):
        sites = np.empty_like(self.sites)
        for idx, tensor in np.ndenumerate(self.sites):
            sites[idx] = tensor.copy()
        horizontal_bonds = np.empty_like(self.horizontal_bonds)
        for idx, tensor in np.ndenumerate(self.horizontal_bonds):
            horizontal_bonds[idx] = tensor.copy()
        vertical_bonds = np.empty_like(self.vertical_bonds)
        for idx, tensor in np.ndenumerate(self.vertical_bonds):
            vertical_bonds[idx] = tensor.copy()
        return XPEPS(sites, horizontal_bonds, vertical_bonds, self.backend)

    def conjugate(self):
        sites = np.empty_like(self.sites)
        for idx, tensor in np.ndenumerate(self.sites):
            sites[idx] = tensor.conj()
        horizontal_bonds = np.empty_like(self.horizontal_bonds)
        for idx, tensor in np.ndenumerate(self.horizontal_bonds):
            horizontal_bonds[idx] = tensor.conj()
        vertical_bonds = np.empty_like(self.vertical_bonds)
        for idx, tensor in np.ndenumerate(self.vertical_bonds):
            vertical_bonds[idx] = tensor.conj()
        return XPEPS(sites, horizontal_bonds, vertical_bonds, self.backend)

    def apply_gate(self, gate, threshold=None, maxrank=None):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        self.apply_operator(tensor, gate.qubits, threshold=threshold, maxrank=maxrank)

    def apply_circuit(self, gates, threshold=None, maxrank=None):
        for gate in gates:
            self.apply_gate(gate, threshold=threshold, maxrank=maxrank)

    def apply_operator(self, operator, sites, threshold=None, maxrank=None):
        operator = self.backend.astensor(operator)
        positions = [divmod(site, self.ncol) for site in sites]
        if len(positions) == 1:
            update.apply_single_site_operator(self, operator, positions[0])
        elif len(positions) == 2 and is_two_local(*positions):
            update.apply_local_pair_operator(self, operator, positions, threshold, maxrank)
        else:
            raise ValueError('nonlocal operator is not supported')

    def __add__(self, other):
        if isinstance(other, XPEPS):
            return self.add(other)
        else:
            return NotImplemented

    def __imul__(self, a):
        if isinstance(a, Number):
            multiplier = a ** (1/(self.nrow * self.ncol))
            for idx in np.ndindex(*self.shape):
                self.sites[idx] *= multiplier
            return self
        else:
            return NotImplemented

    def __itruediv__(self, a):
        if isinstance(a, Number):
            divider = a ** (1/(2 * self.nrow * self.ncol - self.nrow - self.ncol))
            for idx in np.ndindex(self.nrow, self.ncol-1):
                self.horizontal_bonds[idx] /= divider
            for idx in np.ndindex(self.nrow-1, self.ncol):
                self.vertical_bonds[idx] /= divider
            return self
        else:
            return NotImplemented

    def norm(self):
        return sqrt(np.real_if_close(self.inner(self)))

    def add(self, other):
        """
        Add two XPEPS of the same shape and return the sum as a third XPEPS also with the same shape.
        """
        if self.shape != other.shape:
            raise ValueError(f'XPEPS shapes do not match: {self.shape} != {other.shape}')
        sites = np.empty_like(self.sites)
        for i, j in np.ndindex(*self.sites.shape):
            internal_bonds = []
            external_bonds = [4]
            (external_bonds if i == 0 else internal_bonds).append(0)
            (external_bonds if j == 0 else internal_bonds).append(1)
            (external_bonds if i == self.shape[0] - 1 else internal_bonds).append(2)
            (external_bonds if j == self.shape[1] - 1 else internal_bonds).append(3)
            sites[i, j] = tn_add(self.backend, self.sites[i, j], other.sites[i, j], internal_bonds, external_bonds)
        horizontal_bonds = np.empty_like(self.horizontal_bonds)
        for i, j in np.ndindex(self.nrow, self.ncol-1):
            horizontal_bonds[i, j] = tn_add(self.backend, self.horizontal_bonds[i, j], other.horizontal_bonds[i, j], [0], [])
        vertical_bonds = np.empty_like(self.vertical_bonds)
        for i, j in np.ndindex(self.nrow-1, self.ncol):
            vertical_bonds[i, j] = tn_add(self.backend, self.vertical_bonds[i, j], other.vertical_bonds[i, j], [0], [])
        return XPEPS(sites, horizontal_bonds, vertical_bonds, self.backend)

    def amplitude(self, indices):
        if len(indices) != self.nsite:
            raise ValueError('indices number and sites number do not match')
        indices = np.array(indices).reshape(*self.shape)
        sites = np.empty_like(self.sites, dtype=object)
        zero = self.backend.astensor(np.array([1,0], dtype=complex))
        one = self.backend.astensor(np.array([0,1], dtype=complex))
        for i, j in np.ndindex(*self.shape):
            sites[i, j] = self.backend.einsum('ijklx,x->ijkl', self.sites[i,j], one if indices[i,j] else zero)
        return contract.to_value(XPEPS(sites, self.horizontal_bonds, self.vertical_bonds, self.backend))

    def probability(self, indices):
        return np.abs(self.amplitude(indices))**2

    def expectation(self, observable, use_cache=False):
        if use_cache:
            # return self._expectation_with_cache(observable)
            raise NotImplementedError('Expectation with cache is not supported in XPEPS')
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            e += np.real_if_close(self.inner(other))
        return e

    # def _expectation_with_cache(self, observable):
    #     env = contract.create_env_cache(self)
    #     e = 0
    #     for tensor, sites in observable:
    #         other = self.copy()
    #         other.apply_operator(self.backend.astensor(tensor), sites)
    #         rows = [site // self.ncol for site in sites]
    #         up, down = min(rows), max(rows)
    #         e += np.real_if_close(contract.inner_with_env(
    #             other.sites[up:down+1], self.sites[up:down+1], env, up, down
    #         ))
    #     return e

    def contract(self):
        return contract.to_statevector(self)

    def inner(self, other):
        return contract.inner(self, other)

    def statevector(self):
        from .. import statevector
        return statevector.StateVector(self.contract(), self.backend)


def tn_add(backend, a, b, internal_bonds, external_bonds):
    """
    Helper function for addition of two tensor network states with the same structure. 
    Add two site from two tensor network states respecting specified inner and external bond structure. 
    """
    ndim = a.ndim
    shape_a = np.array(np.shape(a))
    shape_b = np.array(np.shape(b))
    shape_c = np.copy(shape_a)
    shape_c[internal_bonds] += shape_b[internal_bonds]
    lim = np.copy(shape_a).astype(object)
    lim[external_bonds] = None    
    a_ind = tuple([slice(lim[i]) for i in range(ndim)])
    b_ind = tuple([slice(lim[i], None) for i in range(ndim)])
    c = backend.zeros(shape_c, dtype=a.dtype)
    c[a_ind] += a
    c[b_ind] += b
    return c


def is_two_local(p, q):
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    return dx == 1 and dy == 0 or dx == 0 and dy == 1
