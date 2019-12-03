import numpy as np
import tensorbackends

from .peps import PEPS

def computational_zeros(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.array([1,0],dtype=complex).reshape(1,1,1,1,2))
    return PEPS(grid, backend)


def computational_ones(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.array([0,1],dtype=complex).reshape(1,1,1,1,2))
    return PEPS(grid, backend)


def computational_basis(nrow, ncol, bits, backend='numpy'):
    backend = tensorbackends.get(backend)
    bits = np.asarray(bits).reshape(nrow, ncol)
    grid = np.empty_like(bits, dtype=object)
    for i, j in np.ndindex(*bits.shape):
        grid[i, j] = backend.astensor(
            np.array([0,1] if bits[i,j] else [1,0],dtype=complex).reshape(1,1,1,1,2)
        )
    return PEPS(grid, backend)
