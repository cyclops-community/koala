import numpy as np 
from .peps import PEPS
from ..backends import get_backend

def tn_add(a, b, internal_bonds, external_bonds):
    r"""
    Helper function for addition of two tensor network states with the same structure. 
    Add two site from two tensor network states respecting specified inner and external bond structure. 
    """
    print(a.shape, b.shape, internal_bonds, external_bonds)
    ndim = np.ndim(a)
    shape_a = np.array(np.shape(a))
    shape_b = np.array(np.shape(b))
    shape_c = np.copy(shape_a)
    shape_c[internal_bonds] += shape_b[internal_bonds]
    c = np.zeros(shape_c, dtype=complex)

    lim = np.copy(shape_a).astype(object)
    lim[external_bonds] = None
    
    ind = tuple([slice(lim[i]) for i in range(ndim)])
    c[ind] += a
    ind = tuple([slice(lim[i], None) for i in range(ndim)])
    c[ind] += b

    return c

def peps_add(peps1, peps2):
    r"""
    Add two PEPS of the same grid shape and return the sum as a third PEPS also with the same grid shape.
    """
    nrows, ncols = peps1.shape
    peps3 = PEPS.empty_state(nrows, ncols, peps1.backend, peps1.threshold, peps1.rescale)
    for r in range(nrows):
        for c in range(ncols):
            internal_bonds = [0,1,2,3]
            external_bonds = [4]
            if c == 0: # left boundary
                internal_bonds.remove(1)
                external_bonds.append(1)
            if c == nrows-1: # right boundary
                internal_bonds.remove(3)
                external_bonds.append(3)
            if r == 0: # upper boundary
                internal_bonds.remove(0)
                external_bonds.append(0)
            if r == ncols-1: # lower boundary
                internal_bonds.remove(2)
                external_bonds.append(2)
            print(peps1[r,c])
            peps3[r,c] = tn_add(peps1[r,c], peps2[r,c], internal_bonds, external_bonds)
    return peps3
            

# test
if __name__=='__main__':
    # test tn_add with two-site mpo addition
    a1 = np.random.random((1,3,4))
    a2 = np.random.random((1,3,7))
    b1 = np.random.random((4,5,1))
    b2 = np.random.random((7,5,1))
    c = np.random.random((3,5))
    a = tn_add(a1, a2, [2], [0,1])
    b = tn_add(b1, b2, [0], [1,2])
    test = np.einsum('dab,bcd,ac->d', a, b ,c)
    ref =  np.einsum('dab,bcd,ac->d', a1, b1,c) + np.einsum('dab,bcd,ac->d', a2, b2,c)
    assert np.allclose(test,ref)
    
    # test peps_add
    be = get_backend('numpy')
    nrow = 3
    ncol = 3
    peps1 = PEPS.zeros_state(nrow, ncol, be)
    peps2 = PEPS.ones_state(nrow, ncol, be)
    print(peps_add(peps1, peps2))