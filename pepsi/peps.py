import random

import numpy as np

from .contraction import contract_peps
from .linear_algebra import einsvd


class PEPS(object):
    def __init__(self, grid):
        self.grid = grid

    @staticmethod
    def zero_states(nrow, ncol):
        grid = np.empty((nrow, ncol), dtype=object)
        for i, j in np.ndindex(nrow, ncol):
            grid[i, j] = np.array([1,0],dtype=complex).reshape([1,1,1,1,2])
        return PEPS(grid)

    @staticmethod
    def one_states(nrow, ncol):
        grid = np.empty((nrow, ncol), dtype=object)
        for i, j in np.ndindex(nrow, ncol):
            grid[i, j] = np.array([0,1],dtype=complex).reshape([1,1,1,1,2])
        return PEPS(grid)

    @staticmethod
    def bit_states(bits):
        bits = np.asarray(bits)
        if bits.ndim != 2:
            raise ValueError('Initial bits must be a 2-d array')
        grid = np.empty(bits.shape)
        for i, j in np.ndindex(nrow, ncol):
            grid[i, j] = np.array([0,1] if bits[i,j] else [1,0],dtype=complex).reshape([1,1,1,1,2])
        return PEPS(grid)

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    def __getitem__(self, key):
        return self.grid[key]

    def __setitem__(self, key, value):
        self.grid[key] = value

    def apply_operator(self, tensor, positions):
        if len(positions) == 1:
            self.apply_operator_one(tensor, positions[0])
        elif len(positions) == 2 and is_two_local(*positions):
            self.apply_operator_two_local(tensor, positions)
        else:
            raise ValueError()

    def apply_operator_one(self, tensor, position):
        """
        Apply a single qubit gate at given position.
        """
        self.grid[position] = np.einsum('ijklx,xy->ijkly', self.grid[position], tensor)

    def apply_operator_two_local(self, tensor, positions):
        """
        Apply a two qubit gate to given positions.
        """
        assert len(positions) == 2
        sites = [self.grid[p] for p in positions]
        prod = tensor

        # contract sites into gate tensor
        site_inds = [*range(5)]
        gate_inds = [*range(4,4+4)]
        result_inds = [*range(4), *range(5,8)]
        prod = np.einsum(sites[0], site_inds, prod, gate_inds, result_inds)

        link0, link1 = get_link(positions[0], positions[1])
        gate_inds = [*range(7)]
        site_inds = [*range(7, 7+4),4]
        site_inds[link1]=link0

        middle = [*range(7, 7+link1), *range(link1+8, 7+4)]
        left = [*range(link0), *range(link0+1,4)]
        right = [*range(5, 7)]
        result_inds = left + middle + right

        prod = np.einsum(sites[1], site_inds, prod, gate_inds, result_inds)

        #svd split sites
        # TODO: allow truncations
        u, sv = einsvd(prod, [0,1,2,6])
        u_inds = [*range(link0)]+[*range(link0+1,4)]+[4]+[link0]
        u_perm = np.argsort(u_inds)
        u = np.transpose(u, u_perm)

        sv_inds = [link1]+ [*range(link1)]+[*range(link1+1,4)]+ [4]
        sv_perm = np.argsort(sv_inds)
        sv = np.transpose(sv, sv_perm)

        self.grid[positions[0]] = u
        self.grid[positions[1]] = sv

    def measure(self, positions):
        result = self.peak(positions, 1)[0]
        for pos, val in zip(positions, result):
            self.apply_single_qubit(np.array([[1-val,0],[0,val]]), pos)
        return result

    def peak(self, positions, nsample):
        prob = contract_peps(self.grid)
        np.absolute(prob, out=prob) # to save memory
        prob **= 2 # to save memory
        ndigits = len(prob)
        to_binary = lambda n: np.array([int(d) for d in f'{n:0{ndigits}b}'])
        positions_array = [i*self.ncol+j for i, j in positions]
        return [to_binary(n)[positions_array] for n in random.choices(range(len(prob)), weights=prob, k=nsample)]

    def contract(self):
        return contract_peps(self.grid)


def get_link(pos1, pos2):
    y1,x1 = pos1
    y2,x2 = pos2
    x = x2-x1
    y = y2-y1
    if x == 0:
        if y == 1:
            return (2,0)
        elif y == -1:
            return (0,2)
        else:
            ValueError("No link between these two positions")
    elif y == 0:
        if x == 1:
            return (3,1)
        elif x == -1:
            return (1,3)
        else: 
            ValueError("No link between these two positions")
    else:
        ValueError("No link between these two positions")


def is_two_local(p, q):
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    return dx == 1 and dy == 0 or dx == 0 and dy == 1
