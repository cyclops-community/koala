from tensorbackends.interface import ImplicitRandomizedSVD

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
    
    if state.using_ctf:
        import ctf
        timer = ctf.timer('einsumsvd_implicit_rand')
        timer.start()
    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, operator, option=ImplicitRandomizedSVD(rank=maxrank))
    if state.using_ctf:
        timer.stop()

    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


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
