import numpy as np 

def tn_add(a, b, internal_bonds, external_bonds):
    r"""
    Helper function for addition of two tensor network states with the same structure. 
    Add two site from two tensor network states respecting specified inner and external bond structure. 
    """
    ndim = np.ndim(a)
    shape_a = np.array(np.shape(a))
    shape_b = np.array(np.shape(b))
    shape_c = np.copy(shape_a)
    shape_c[internal_bonds] += shape_b[internal_bonds]
    c = np.zeros(shape_c)

    lim = np.copy(shape_a).astype(object)
    lim[external_bonds] = None
    
    ind = tuple([slice(lim[i]) for i in range(ndim)])
    c[ind] = a
    ind = tuple([slice(lim[i], None) for i in range(ndim)])
    c[ind] = b

    return c


# test
if __name__=='__main__':
    # test two side mpo addition
    a1 = np.random.random((3,4))
    a2 = np.random.random((3,7))
    b1 = np.random.random((4,5))
    b2 = np.random.random((7,5))
    c = np.random.random((3,5))
    a = tn_add(a1, a2, [1], [0])
    b = tn_add(b1, b2, [0], [1])
    test = np.einsum('ab,bc,ac', a, b ,c)
    ref =  np.einsum('ab,bc,ac', a1, b1,c) + np.einsum('ab,bc,ac', a2, b2,c)
    assert np.isclose(test,ref)