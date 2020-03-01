
def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklxp,xy->ijklyp', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, threshold, maxrank, randomized_svd):
    assert len(positions) == 2
    if maxrank is not None and randomized_svd:
        apply_local_pair_operator_randomized_svd(state, operator, positions, threshold, maxrank)
    else:
        apply_local_pair_operator_direct_svd(state, operator, positions, threshold, maxrank)


def apply_local_pair_operator_direct_svd(state, operator, positions, threshold, maxrank):
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        prod_subscripts = 'abcdxp,cfghyq,xyuv->abndup,nfghvq'
        scale_u_subscripts = 'absdup,s->absdup'
        scale_v_subscripts = 'sbcdvp,s->sbcdvp'
        link = (2, 0)
    elif x_pos[0] > y_pos[0]: # [y x]^T
        prod_subscripts = 'abcdxp,efahyq,xyuv->nbcdup,efnhvq'
        link = (0, 2)
        scale_u_subscripts = 'sbcdup,s->sbcdup'
        scale_v_subscripts = 'absdvp,s->absdvp'
    elif x_pos[1] < y_pos[1]: # [x y]
        prod_subscripts = 'abcdxp,efgbyq,xyuv->ancdup,efgnvq'
        link = (1, 3)
        scale_u_subscripts = 'ascdup,s->ascdup'
        scale_v_subscripts = 'abcsvp,s->abcsvp'
    elif x_pos[1] > y_pos[1]: # [y x]
        prod_subscripts = 'abcdxp,edghyq,xyuv->abcnup,enghvq'
        link = (3, 1)
        scale_u_subscripts = 'abcsup,s->abcsup'
        scale_v_subscripts = 'ascdvp,s->ascdvp'
    else:
        assert False

    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, operator)
    u, s, v = truncate(state.backend, u, s, v, link[0], link[1], threshold=threshold, maxrank=maxrank)
    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


def apply_local_pair_operator_randomized_svd(state, operator, positions, threshold, maxrank, niter=1, oversamp=5):
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    # split the operator
    x_operator, s, y_operator = state.backend.einsumsvd('xyuv->xuA,yvA', operator)
    x_operator, s, y_operator = truncate(state.backend, x_operator, s, y_operator, 2, 2, threshold=1e-5)
    s = s ** 0.5
    x_operator = state.backend.einsum('xuA,A->xuA', x_operator, s)
    y_operator = state.backend.einsum('yvA,A->yvA', y_operator, s)

    if x_pos[0] < y_pos[0]: # [x y]^T
        apply_on_x = 'abcdxp,xuA->(abdup)(cA)'
        apply_on_y = 'cfghyp,yvA->(cA)(fghvp)'
        m_axes = [0, 1, 3, 4, 5]
        n_axes = [1, 2, 3, 4, 5]
        extract_x = 'abdupc,c->abcdup'
        extract_y = 'cfghvp,c->cfghvp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        apply_on_x = 'abcdxp,xuA->(bcdup)(aA)'
        apply_on_y = 'efahyp,yvA->(aA)(efhvp)'
        m_axes = [1, 2, 3, 4, 5]
        n_axes = [0, 1, 3, 4, 5]
        extract_x = 'bcdupa,a->abcdup'
        extract_y = 'aefhvp,a->efahvp'
    elif x_pos[1] < y_pos[1]: # [x y]
        apply_on_x = 'abcdxp,xuA->(acdup)(bA)'
        apply_on_y = 'efgbyp,yvA->(bA)(efgvp)'
        m_axes = [0, 2, 3, 4, 5]
        n_axes = [0, 1, 2, 4, 5]
        extract_x = 'acdupb,b->abcdup'
        extract_y = 'befgvp,b->efgbvp'
    elif x_pos[1] > y_pos[1]: # [y x]
        apply_on_x = 'abcdxp,xuA->(abcup)(dA)'
        apply_on_y = 'edghyp,yvA->(dA)(eghvp)'
        m_axes = [0, 1, 2, 4, 5]
        n_axes = [0, 2, 3, 4, 5]
        extract_x = 'abcupd,d->abcdup'
        extract_y = 'deghvp,d->edghvp'
    else:
        assert False

    m_shape = [x.shape[d] for d in m_axes]
    n_shape = [y.shape[d] for d in n_axes]

    x_mat = state.backend.einsum(apply_on_x, x, x_operator)
    y_mat = state.backend.einsum(apply_on_y, y, y_operator)

    u, s, vh = randomized_svd(state.backend, x_mat, y_mat, maxrank, niter, oversamp)
    s = s ** 0.5
    x = state.backend.einsum(extract_x, u.reshape(*m_shape, -1), s)
    y = state.backend.einsum(extract_y, vh.reshape(-1, *n_shape), s)

    state.grid[x_pos] = x
    state.grid[y_pos] = y


def randomized_svd(backend, a, b, rank, niter=1, oversamp=5):
    m, n = a.shape[0], b.shape[1]
    assert a.shape[1] == b.shape[0]
    r = min(rank + oversamp, m, n, a.shape[1])
    # find subspace
    q = backend.random.uniform(low=-1.0, high=1.0, size=(n, r)).astype(complex)
    a_H, b_H = a.H, b.H
    for i in range(niter):
        q = b_H @ (a_H @ (a @ (b @ q)))
        q, _ = backend.qr(q)
    q = a @ (b @ q)
    q, _ = backend.qr(q)
    # svd
    ab_sub = (q.H @ a) @ b
    u_sub, s, vh = backend.svd(ab_sub)
    u = q @ u_sub
    if rank < r:
        u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
    return u, s, vh


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
