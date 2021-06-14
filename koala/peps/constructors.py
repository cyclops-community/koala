import numpy as np
import tensorbackends

from .peps import PEPS

def computational_zeros(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.array([1,0],dtype=complex).reshape(1,1,1,1,2,1))
    return PEPS(grid, backend)


def computational_ones(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.array([0,1],dtype=complex).reshape(1,1,1,1,2,1))
    return PEPS(grid, backend)


def computational_basis(nrow, ncol, bits, backend='numpy'):
    backend = tensorbackends.get(backend)
    bits = np.asarray(bits).reshape(nrow, ncol)
    grid = np.empty_like(bits, dtype=object)
    for i, j in np.ndindex(*bits.shape):
        grid[i, j] = backend.astensor(
            np.array([0,1] if bits[i,j] else [1,0],dtype=complex).reshape(1,1,1,1,2,1)
        )
    return PEPS(grid, backend)


def random(nrow, ncol, rank, physical_dim=2, dual_dim=1, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        shape = (
            rank if i > 0 else 1,
            rank if j < ncol - 1 else 1,
            rank if i < nrow - 1 else 1,
            rank if j > 0 else 1,
            physical_dim, dual_dim,
        )
        grid[i, j] = backend.random.uniform(-1,1,shape) + 1j * backend.random.uniform(-1,1,shape)
    return PEPS(grid, backend)


def identity(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.eye(2,dtype=complex).reshape(1,1,1,1,2,2))
    return PEPS(grid, backend)
