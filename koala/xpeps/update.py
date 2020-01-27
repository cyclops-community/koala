from string import ascii_letters as chars


def apply_single_site_operator(state, operator, position):
    state.sites[position] = state.backend.einsum('ijklx,xy->ijkly', state.sites[position], operator)


def apply_local_pair_operator(state, operator, positions, threshold, maxrank):
    assert len(positions) == 2
    backend = state.backend
    x_pos, y_pos = positions
    x, y = state.sites[x_pos], state.sites[y_pos]

    def absorb_bonds(site, i, j, directions):
        nonlocal state
        if 'above' in directions and i > 0:
            site = state.backend.einsum('abcdx,a->abcdx', site, state.vertical_bonds[i-1, j])
        if 'below' in directions and i < state.nrow - 1:
            site = state.backend.einsum('abcdx,c->abcdx', site, state.vertical_bonds[i, j])
        if 'left' in directions and j > 0:
            site = state.backend.einsum('abcdx,b->abcdx', site, state.horizontal_bonds[i, j-1])
        if 'right' in directions and j < state.ncol - 1:
            site = state.backend.einsum('abcdx,d->abcdx', site, state.horizontal_bonds[i, j])
        return site

    def absorb_bonds_inv(site, i, j, directions):
        nonlocal state
        if 'above' in directions and i > 0:
            site = state.backend.einsum('abcdx,a->abcdx', site, 1/state.vertical_bonds[i-1, j])
        if 'below' in directions and i < state.nrow - 1:
            site = state.backend.einsum('abcdx,c->abcdx', site, 1/state.vertical_bonds[i, j])
        if 'left' in directions and j > 0:
            site = state.backend.einsum('abcdx,b->abcdx', site, 1/state.horizontal_bonds[i, j-1])
        if 'right' in directions and j < state.ncol - 1:
            site = state.backend.einsum('abcdx,d->abcdx', site, 1/state.horizontal_bonds[i, j])
        return site

    if x_pos[0] < y_pos[0]: # [x y]^T
        x = absorb_bonds(x, *x_pos, {'above', 'left', 'right'})
        y = absorb_bonds(y, *y_pos, {'below', 'left', 'right'})
        w = state.vertical_bonds[x_pos[0], x_pos[1]]
        u, s, v = backend.einsumsvd('abcdx,c,cfghy,xyuv->abndu,nfghv', x, w, y, operator)
        u, s, v = truncate(backend, u, s, v, u_axis=2, v_axis=0, threshold=threshold, maxrank=maxrank)
        state.sites[x_pos] = absorb_bonds_inv(u, *x_pos, {'above', 'left', 'right'})
        state.sites[y_pos] = absorb_bonds_inv(v, *y_pos, {'below', 'left', 'right'})
        state.vertical_bonds[x_pos[0], x_pos[1]] = s.astype(complex)
    elif x_pos[0] > y_pos[0]: # [y x]^T
        x = absorb_bonds(x, *x_pos, {'below', 'left', 'right'})
        y = absorb_bonds(y, *y_pos, {'above', 'left', 'right'})
        w = state.vertical_bonds[y_pos[0], x_pos[1]]
        u, s, v = backend.einsumsvd('abcdx,a,efahy,xyuv->nbcdu,efnhv', x, w, y, operator)
        u, s, v = truncate(backend, u, s, v, u_axis=0, v_axis=2, threshold=threshold, maxrank=maxrank)
        state.sites[x_pos] = absorb_bonds_inv(u, *x_pos, {'below', 'left', 'right'})
        state.sites[y_pos] = absorb_bonds_inv(v, *y_pos, {'above', 'left', 'right'})
        state.vertical_bonds[y_pos[0], x_pos[1]] = s.astype(complex)
    elif x_pos[1] < y_pos[1]: # [x y]
        x = absorb_bonds(x, *x_pos, {'above', 'below', 'left'})
        y = absorb_bonds(y, *y_pos, {'above', 'below', 'right'})
        w = state.horizontal_bonds[x_pos[0], x_pos[1]]
        u, s, v = backend.einsumsvd('abcdx,d,edghy,xyuv->abcnu,enghv', x, w, y, operator)
        u, s, v = truncate(backend, u, s, v, u_axis=3, v_axis=1, threshold=threshold, maxrank=maxrank)
        state.horizontal_bonds[x_pos[0], x_pos[1]] = s.astype(complex)
        state.sites[x_pos] = absorb_bonds_inv(u, *x_pos, {'above', 'below', 'left'})
        state.sites[y_pos] = absorb_bonds_inv(v, *y_pos, {'above', 'below', 'right'})
    elif x_pos[1] > y_pos[1]: # [y x]
        x = absorb_bonds(x, *x_pos, {'above', 'below', 'right'})
        y = absorb_bonds(y, *y_pos, {'above', 'below', 'left'})
        w = state.horizontal_bonds[x_pos[0], y_pos[1]]
        u, s, v = backend.einsumsvd('abcdx,b,efgby,xyuv->ancdu,efgnv', x, w, y, operator)
        u, s, v = truncate(backend, u, s, v, u_axis=1, v_axis=3, threshold=threshold, maxrank=maxrank)
        state.sites[x_pos] = absorb_bonds_inv(u, *x_pos, {'above', 'below', 'right'})
        state.sites[y_pos] = absorb_bonds_inv(v, *y_pos, {'above', 'below', 'left'})
        state.horizontal_bonds[x_pos[0], y_pos[1]] = s.astype(complex)
    else:
        assert False


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
