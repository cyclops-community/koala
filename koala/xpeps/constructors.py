import numpy as np
import tensorbackends

from .xpeps import XPEPS

def computational_zeros(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    sites = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        sites[i, j] = backend.astensor(np.array([1,0],dtype=complex).reshape(1,1,1,1,2))
    horizontal_bonds = _identity_bonds(nrow, ncol-1, backend)
    vertical_bonds = _identity_bonds(nrow-1, ncol, backend)
    return XPEPS(sites, horizontal_bonds, vertical_bonds, backend)


def computational_ones(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    sites = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        sites[i, j] = backend.astensor(np.array([0,1],dtype=complex).reshape(1,1,1,1,2))
    horizontal_bonds = _identity_bonds(nrow, ncol-1, backend)
    vertical_bonds = _identity_bonds(nrow-1, ncol, backend)
    return XPEPS(sites, horizontal_bonds, vertical_bonds, backend)


def computational_basis(nrow, ncol, bits, backend='numpy'):
    backend = tensorbackends.get(backend)
    bits = np.asarray(bits).reshape(nrow, ncol)
    sites = np.empty_like(bits, dtype=object)
    for i, j in np.ndindex(*bits.shape):
        sites[i, j] = backend.astensor(
            np.array([0,1] if bits[i,j] else [1,0],dtype=complex).reshape(1,1,1,1,2)
        )
    horizontal_bonds = _identity_bonds(nrow, ncol-1, backend)
    vertical_bonds = _identity_bonds(nrow-1, ncol, backend)
    return XPEPS(sites, horizontal_bonds, vertical_bonds, backend)


def _identity_bonds(nrow, ncol, backend):
    bonds = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        bonds[i, j] = backend.ones(1, dtype=complex)
    return bonds
