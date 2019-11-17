import numpy as np 

def tn_add(A, B, inner_bonds, external_bonds):
    r"""
    Helper function for addition of two tensor network states with the same structure. 
    Add two site from two tensor network states respecting specified inner and external bond structure. 
    """
    ndim = np.ndim(A)
    shapeA = np.array(np.shape(A))
    shapeB = np.array(np.shape(B))
    shapeC = np.copy(shapeA)
    shapeC[inner_bonds] += shapeB[inner_bonds]
    C = np.zeros(shapeC)

    lim = np.copy(shapeA).astype(object)
    lim[external_bonds] = None
    
    ind = tuple([slice(lim[i]) for i in range(ndim)])
    C[ind] = A
    ind = tuple([slice(lim[i], None) for i in range(ndim)])
    C[ind] = B

    return C


# test
if __name__=='__main__':
    # test two side mpo addition
    A1 = np.random.random((3,4))
    A2 = np.random.random((3,7))
    B1 = np.random.random((4,5))
    B2 = np.random.random((7,5))
    C = np.random.random((3,5))
    A = tn_add(A1, A2, [1], [0])
    B = tn_add(B1, B2, [0], [1])
    test = np.einsum('ab,bc,ac', A, B,C)
    ref =  np.einsum('ab,bc,ac', A1, B1,C) + np.einsum('ab,bc,ac', A2, B2,C)
    assert np.isclose(test,ref)