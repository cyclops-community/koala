import numpy as np


def einsvd(A, inds, **kwargs):
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
    B = np.moveaxis(A, inds, [*range(len(inds))])
    left_dim = np.prod(B.shape[:len(inds)])
    shape = B.shape
    B = np.reshape(B, (left_dim, -1))
    u,s,v = np.linalg.svd(B, full_matrices=False)
    s = truncate(s, **kwargs)[0]
    u = np.reshape(u[:,:len(s)], (*shape[:len(inds)], -1))
    sv = np.reshape(np.diag(s)@v[:len(s), :], (-1, *shape[len(inds):]))
    return u, sv

def truncate(s, **kwargs): #cutoff = 0.0, mode = "abs",maxdim = -1, mindim = 1
    """
        truncate(s,cutoff = 0.0, mode = "abs",maxdim = -1, mindim = 1)
    Truncate array `s` according to given rule. maxdim/ mindim has higher priority than absolute/relative cutoff.
    # Parameters: 
    - `s`: array (of singular values)
    - `cutoff`: singular values smaller than this value will be truncated (actual operation depend on mode)
    - `mode`: define truncation mode. 
        "abs" : throw away all singular values < `cutoff`
        "rel" : `cutoff` in [0,1] is interpreted as a percentage (e.g. cutoff=0.5 in "rel" mode means throw away small singular values s.t. norm(remaining vector) = 0.5* norm(original vector))
    - `maxdim`: regardless of other rules, number of sigular values are capped at `maxdim`
    - `mindim`: regardless of other rules, will keep at least `mindim` singular values
    """
    cutoff = kwargs.get('cutoff', 0.0)
    mode = kwargs.get('mode', "abs")
    maxdim = kwargs.get('maxdim', len(s))
    mindim = kwargs.get('mindim', 1)

    p = np.argsort(s)[::-1]
    s = s[p]
    s = np.clip(s, 0.0, None) # zero out negative singular values
    s_trunc = s
    if mode == "abs" and cutoff>0.0:
        s_trunc = s[s<=cutoff]
    elif mode == "rel":
        cutoff_val = sum(s**2)*(1-cutoff)
        cut = 0.0
        ind = len(s)-1
        while True:
            cut += s[ind]**2
            if cut > cutoff_val:
                break
            else:
                ind-=1
        s_trunc = np.resize(s, (ind+2))
    else:
        ValueError("Unknown truncation mode")
    if len(s_trunc) < mindim:
        s_trunc = s[:mindim+1]
    if len(s_trunc) > maxdim:
        s_trunc = s[:maxdim]
    truncated_n = len(s) - len(s_trunc)
    return s_trunc, truncated_n

if __name__ == "__main__":
    # a simple test for einsvd	
    dims = (np.random.randint(1,10, size=5)).tolist()
    A = np.random.random_sample(dims)
    u, sv = einsvd(A, [2,4])
    print(np.shape(u))
    product = np.einsum(u, [1,2,3], sv, [3,4,5,6])
    assert np.allclose(product, np.transpose(A, [2,4,0,1,3]))
    # test truncation dim match
    u, sv = einsvd(A, [2,4], maxdim = 1)
    print(np.shape(u))
    product = np.einsum(u, [1,2,3], sv, [3,4,5,6])
    assert np.shape(product) == np.shape(np.transpose(A, [2,4,0,1,3])) 
