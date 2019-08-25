
def einsvd(A, inds):
    """
    Do svd on tensor A with indices specified indices in u and others in sv.
    Indices in u are in the order provided in `inds`.
    Indices in sv are in the same ordering as in original tensor A. 

    # Arguments:
    A - multi dimensional array
    inds - list of indices that will go to u after svd

    # Example:
    >>> A = np.random.rand(1,2,3,4,5,6,7)
    >>> u, sv = einsvd(A, [2,4,6])
    >>> u.shape
    (3, 5, 7, 48)
    >>> sv.shape
    (48, 1, 2, 4, 6)
    """
    B = np.moveaxis(A, inds, [i for i in range(len(inds))])
    left_dim = np.prod(B.shape[:len(inds)])
    shape = B.shape
    B = np.reshape(B, (left_dim, -1))
    u,s,v = np.linalg.svd(B, full_matrices=False)
    u = np.reshape(u, shape[:len(inds)]+(-1,))
    sv = np.reshape(np.diag(s)@v, (-1,)+shape[len(inds):])
    return u, sv


if __name__ == "__main__":
    import numpy as np
    # a simple test for einsvd	
    dims = (np.random.randint(1,10, size=5)).tolist()
    A = np.random.random_sample(dims)
    u, sv = einsvd(A, [2,4])
    product = np.einsum(u, [1,2,3], sv, [3,4,5,6])
    assert(np.allclose(product, np.transpose(A, [2,4,0,1,3])))