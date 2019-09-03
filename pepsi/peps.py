import random

import peps


class Peps(object):
    def __init__(self, grid = None, row = None, col = None):
        if grid == None:
            if row == None or col ==None:
                ValueError("Either the grid or the dimensions shoule be provided to construct a PEPs")
            else:
                self.grid = np.empty((row, col), dtype=object)
                for i in range(row):
                    for j in range(col):
                        self.grid[i,j] = np.array([1,0],dtype=complex).reshape([1,1,1,1,2])
        else:
            self.grid = grid

    def __getitem__(self, key):
        return self.grid[key]

    def __setitem__(self, key, value):
        self.grid[key] = value

    def apply_single_qubit(self, gate_tensor, position):
        """
        Apply a single qubit gate at given position.
        """
        self.grid[position] = np.einsum('ijklx,xy->ijlky', self.grid[position], gate_tensor)

    def apply_two_qubit_local(self, gate_tensor, positions):
        """
        Apply a two qubit gate to given positions (qubits)
        - gate:
        """
        assert len(positions) == 2
        sites = [self.grid[p] for p in positions]
        prod = gate_tensor

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
        prob = _contract(self.grid)
        np.absolute(prob, out=prob) # to save memory
        prob **= 2 # to save memory
        ndigits = len(prob)
        to_binary = lambda n: np.array([int(d) for d in f'{n:0{ndigits}b}'])
        positions_array = [i*self.col+j for i, j in positions]
        return [to_binary(n)[positions_array] for n in random.choices(range(len(prob)), weights=prob, k=nsample)]


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


def _contract(grid):
    grid = grid.copy()
    for i in range(len(grid.shape[0])):
        for j in range(len(grid.shape[1])):
            grid[i, j] = np.transpose(grid, axes=(0,3,2,1,4))
    peps_obj = peps.PEPS(grid, backend='numpy')
    return peps_obj.contract().match_virtual().reshape(-1)
