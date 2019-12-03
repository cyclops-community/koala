"""
This module defines PEPS and operations on it.
"""

import random
from math import sqrt
from numbers import Number
from string import ascii_letters as chars

import numpy as np
import tensorbackends

from ..quantum_state import QuantumState
from ..gates import tensorize
from .contraction import contract_peps, contract_peps_value, contract_inner, create_env_cache, contract_with_env


class PEPS(QuantumState):
    def __init__(self, grid, backend):
        self.backend = tensorbackends.get(backend)
        self.grid = grid

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def nsite(self):
        return self.nrow * self.ncol

    def __getitem__(self, position):
        return self.grid[position]

    def copy(self):
        grid = np.empty_like(self.grid)
        for idx, tensor in np.ndenumerate(self.grid):
            grid[idx] = tensor.copy()
        return PEPS(grid, self.backend)

    def conjugate(self):
        grid = np.empty_like(self.grid)
        for idx, tensor in np.ndenumerate(self.grid):
            grid[idx] = tensor.conj()
        return PEPS(grid, self.backend)

    def apply_gate(self, gate, threshold=None):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        self.apply_operator(tensor, gate.qubits, threshold=threshold)

    def apply_circuit(self, gates, threshold=None):
        for gate in gates:
            self.apply_gate(gate, threshold=threshold)

    def apply_operator(self, operator, sites, threshold=None):
        operator = self.backend.astensor(operator)
        positions = [divmod(site, self.ncol) for site in sites]
        if len(positions) == 1:
            self.apply_operator_one(operator, positions[0])
        elif len(positions) == 2 and is_two_local(*positions):
            self.apply_operator_two_local(operator, positions, threshold)
        else:
            raise ValueError('nonlocal operator is not supported')

    def __add__(self, other):
        if isinstance(other, PEPS):
            return self.add(other)
        else:
            return NotImplemented

    def __imul__(self, a):
        if isinstance(a, Number):
            multiplier = a ** (1/(self.nrow * self.ncol))
            for idx in np.ndindex(*self.shape):
                self.grid[idx] *= multiplier
            return self
        else:
            return NotImplemented

    def __itruediv__(self, a):
        if isinstance(a, Number):
            divider = a ** (1/(self.nrow * self.ncol))
            for idx in np.ndindex(*self.shape):
                self.grid[idx] /= divider
            return self
        else:
            return NotImplemented

    def apply_operator_one(self, operator, position):
        """Apply a single qubit gate at given position."""
        self.grid[position] = self.backend.einsum('ijklx,xy->ijkly', self.grid[position], operator)

    def apply_operator_two_local(self, operator, positions, threshold):
        """Apply a two qubit gate to given positions."""
        assert len(positions) == 2
        sites = [self.grid[p] for p in positions]

        # contract sites into operator
        site_inds = range(5)
        gate_inds = range(4,4+4)
        result_inds = [*range(4), *range(5,8)]

        site_terms = ''.join(chars[i] for i in site_inds)
        gate_terms = ''.join(chars[i] for i in gate_inds)
        result_terms = ''.join(chars[i] for i in result_inds)
        einstr = f'{site_terms},{gate_terms}->{result_terms}'
        prod = self.backend.einsum(einstr, sites[0], operator)

        link0, link1 = get_link(positions[0], positions[1])
        gate_inds = range(7)
        site_inds = [*range(7, 7+4), 4]
        site_inds[link1] = link0

        middle = [*range(7, 7+link1), *range(link1+8, 7+4)]
        left = [*range(link0), *range(link0+1,4)]
        right = range(5, 7)
        result_inds = [*left, *middle, *right]

        site_terms = ''.join(chars[i] for i in site_inds)
        gate_terms = ''.join(chars[i] for i in gate_inds)
        result_terms = ''.join(chars[i] for i in result_inds)
        einstr = f'{site_terms},{gate_terms}->{result_terms}'
        prod = self.backend.einsum(einstr, sites[1], prod)

        # svd split sites
        prod_inds = [*left, *middle, *right]
        u_inds = [*range(link0), link0, *range(link0+1,4), 5]
        v_inds = [*range(7, 7+link1), link0, *range(link1+8, 7+4), 6]
        prod_terms = ''.join(chars[i] for i in prod_inds)
        u_terms = ''.join(chars[i] for i in u_inds)
        v_terms = ''.join(chars[i] for i in v_inds)
        einstr = f'{prod_terms}->{u_terms},{v_terms}'
        u, s, v = self.backend.einsvd(einstr, prod)
        u, s, v = truncate(self.backend, u, s, v, u_inds.index(link0), v_inds.index(link0), threshold=threshold)
        s = s ** 0.5
        u = self.backend.einsum(f'{u_terms},{chars[link0]}->{u_terms}', u, s)
        v = self.backend.einsum(f'{v_terms},{chars[link0]}->{v_terms}', v, s)
        self.grid[positions[0]] = u
        self.grid[positions[1]] = v

    def norm(self):
        return sqrt(np.real_if_close(self.inner(self)))

    def add(self, other):
        """
        Add two PEPS of the same grid shape and return the sum as a third PEPS also with the same grid shape.
        """
        if self.shape != other.shape:
            raise ValueError(f'PEPS shapes do not match: {self.shape} != {other.shape}')
        grid = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(*self.grid.shape):
            internal_bonds = []
            external_bonds = [4]
            (external_bonds if i == 0 else internal_bonds).append(0)
            (external_bonds if j == 0 else internal_bonds).append(1)
            (external_bonds if i == self.shape[0] - 1 else internal_bonds).append(2)
            (external_bonds if j == self.shape[1] - 1 else internal_bonds).append(3)
            grid[i, j] = tn_add(self.backend, self[i, j], other[i, j], internal_bonds, external_bonds)
        return PEPS(grid, self.backend)

    def amplitude(self, indices):
        if len(indices) != self.nsite:
            raise ValueError('indices number and sites number do not match')
        indices = np.array(indices).reshape(*self.shape)
        grid = np.empty_like(self.grid, dtype=object)
        zero = self.backend.astensor(np.array([1,0], dtype=complex))
        one = self.backend.astensor(np.array([0,1], dtype=complex))
        for i, j in np.ndindex(*self.shape):
            grid[i, j] = self.backend.einsum('ijklx,x->ijkl', self.grid[i,j], one if indices[i,j] else zero)
        return contract_peps_value(grid)

    def probability(self, indices):
        return np.abs(self.amplitude(indices))**2

    def expectation(self, observable, use_cache=False):
        if use_cache:
            return self._expectation_with_cache(observable)
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            e += np.real_if_close(self.inner(other))
        return e

    def _expectation_with_cache(self, observable):
        env = create_env_cache(self.grid)
        e = 0
        for tensor, sites in observable:
            other = self.copy()
            other.apply_operator(self.backend.astensor(tensor), sites)
            rows = [site // self.ncol for site in sites]
            up, down = min(rows), max(rows)
            e += np.real_if_close(contract_with_env(
                other.grid[up:down+1], self.grid[up:down+1], env, up, down
            ))
        return e

    def measure(self, positions):
        result = self.peak(positions, 1)[0]
        for pos, val in zip(positions, result):
            self.apply_operator_one(np.array([[1-val,0],[0,val]]), pos)
        return result

    def contract(self):
        return contract_peps(self.grid)

    def inner(self, peps):
        return contract_inner(self.grid, peps.grid)


def get_link(p, q):
    if not is_two_local(p, q):
        raise ValueError(f'No link between {p} and {q}')
    dx, dy = q[0] - p[0], q[1] - p[1]
    if (dx, dy) == (0, 1):
        return (3, 1)
    elif (dx, dy) == (0, -1):
        return (1, 3)
    elif (dx, dy) == (1, 0):
        return (2, 0)
    elif (dx, dy) == (-1, 0):
        return (0, 2)
    else:
        assert False


def is_two_local(p, q):
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    return dx == 1 and dy == 0 or dx == 0 and dy == 1


def truncate(backend, u, s, v, u_axis, v_axis, threshold=None):
    if threshold is None: threshold = 0.0
    residual = backend.norm(s) * threshold
    rank = max(next(r for r in range(s.shape[0], 0, -1) if backend.norm(s[r-1:]) >= residual), 0)
    u_slice = tuple(slice(None) if i != u_axis else slice(rank) for i in range(u.ndim))
    v_slice = tuple(slice(None) if i != v_axis else slice(rank) for i in range(v.ndim))
    s_slice = slice(rank)
    return u[u_slice], s[s_slice], v[v_slice]


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
