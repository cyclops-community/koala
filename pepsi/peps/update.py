from string import ascii_letters as chars

def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklx,xy->ijkly', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, threshold, maxrank):
    assert len(positions) == 2
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        prod_subscripts = 'abcdx,cfghy,xyuv->abndu,nfghv'
        scale_u_subscripts = 'absdu,s->absdu'
        scale_v_subscripts = 'sbcdv,s->sbcdv'
        link = (2, 0)
    elif x_pos[0] > y_pos[0]: # [y x]^T
        prod_subscripts = 'abcdx,efahy,xyuv->nbcdu,efnhv'
        link = (0, 2)
        scale_u_subscripts = 'sbcdu,s->sbcdu'
        scale_v_subscripts = 'absdv,s->absdv'
    elif x_pos[1] < y_pos[1]: # [x y]
        prod_subscripts = 'abcdx,edghy,xyuv->abcnu,enghv'
        link = (3, 1)
        scale_u_subscripts = 'abcsu,s->abcsu'
        scale_v_subscripts = 'ascdv,s->ascdv'
    elif x_pos[1] > y_pos[1]: # [y x]
        prod_subscripts = 'abcdx,efgby,xyuv->ancdu,efgnv'
        link = (1, 3)
        scale_u_subscripts = 'ascdu,s->ascdu'
        scale_v_subscripts = 'abcsv,s->abcsv'
    else:
        assert False

    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, operator)
    u, s, v = truncate(state.backend, u, s, v, link[0], link[1], threshold=threshold)
    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


def apply_low_rank_update(backend, environment, right_side, rank):
    """
    Update sites based on low rank constrained least square:
    Inputs:
        backend: the backend library to be used
        environment: 4-d tensor
        right_side: 4-d tensor
        rank: the rank of each returned site
    Output:
        array including updated sites
    """
    length = environment.shape[0]
    environment = environment.reshape((length * length, length * length))
    right_side = right_side.reshape((length * length, length * length))
    R_inv = backend.inv(backend.transpose(backend.cholesky(environment)))
    B = backend.matmul(right_side, R_inv)
    U, s, VT = backend.svd(B)
    return [
        backend.matmul(U[:, :rank], backend.diag(s[:rank])).reshape(
            (length, length, rank)),
        backend.matmul(R_inv, backend.transpose(VT[:rank, :])).reshape(
            (length, length, rank))
    ]


def truncate(backend, u, s, v, u_axis, v_axis, threshold=None, maxrank=None):
    if threshold is None: threshold = 0.0
    residual = backend.norm(s) * threshold
    rank = max(next(r for r in range(s.shape[0], 0, -1) if backend.norm(s[r-1:]) >= residual), 0)
    if maxrank is not None and rank > maxrank:
        rank = maxrank
    u_slice = tuple(slice(None) if i != u_axis else slice(rank) for i in range(u.ndim))
    v_slice = tuple(slice(None) if i != v_axis else slice(rank) for i in range(v.ndim))
    s_slice = slice(rank)
    return u[u_slice], s[s_slice], v[v_slice]
