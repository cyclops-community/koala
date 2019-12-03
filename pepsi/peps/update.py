from string import ascii_letters as chars

def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklx,xy->ijkly', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, threshold):
    assert len(positions) == 2
    sites = [state.grid[p] for p in positions]

    # contract sites into operator
    site_inds = range(5)
    gate_inds = range(4,4+4)
    result_inds = [*range(4), *range(5,8)]

    site_terms = ''.join(chars[i] for i in site_inds)
    gate_terms = ''.join(chars[i] for i in gate_inds)
    result_terms = ''.join(chars[i] for i in result_inds)
    einstr = f'{site_terms},{gate_terms}->{result_terms}'
    prod = state.backend.einsum(einstr, sites[0], operator)

    link0, link1 = get_link(positions[0], positions[1])
    gate_inds = range(7)
    site_inds = [*range(7, 7+4), 4]
    site_inds[link1] = link0

    middle = [*range(7, 7+link1), *range(link1+8, 7+4)]
    left = [*range(link0), *range(link0+1,4)]
    right = range(5, 7)
    result_inds = [*left, *middle, *right]

    site_terms = ''.join(chars[i] for i in site_inds)
    gate_terms = ''.join(chars[i] for i in gate_inds)
    result_terms = ''.join(chars[i] for i in result_inds)
    einstr = f'{site_terms},{gate_terms}->{result_terms}'
    prod = state.backend.einsum(einstr, sites[1], prod)

    # svd split sites
    prod_inds = [*left, *middle, *right]
    u_inds = [*range(link0), link0, *range(link0+1,4), 5]
    v_inds = [*range(7, 7+link1), link0, *range(link1+8, 7+4), 6]
    prod_terms = ''.join(chars[i] for i in prod_inds)
    u_terms = ''.join(chars[i] for i in u_inds)
    v_terms = ''.join(chars[i] for i in v_inds)
    einstr = f'{prod_terms}->{u_terms},{v_terms}'
    u, s, v = state.backend.einsvd(einstr, prod)
    u, s, v = truncate(state.backend, u, s, v, u_inds.index(link0), v_inds.index(link0), threshold=threshold)
    s = s ** 0.5
    u = state.backend.einsum(f'{u_terms},{chars[link0]}->{u_terms}', u, s)
    v = state.backend.einsum(f'{v_terms},{chars[link0]}->{v_terms}', v, s)
    state.grid[positions[0]] = u
    state.grid[positions[1]] = v


def get_link(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if (dx, dy) == (0, 1):
        return (3, 1)
    elif (dx, dy) == (0, -1):
        return (1, 3)
    elif (dx, dy) == (1, 0):
        return (2, 0)
    elif (dx, dy) == (-1, 0):
        return (0, 2)
    else:
        assert False


def truncate(backend, u, s, v, u_axis, v_axis, threshold=None):
    if threshold is None: threshold = 0.0
    residual = backend.norm(s) * threshold
    rank = max(next(r for r in range(s.shape[0], 0, -1) if backend.norm(s[r-1:]) >= residual), 0)
    u_slice = tuple(slice(None) if i != u_axis else slice(rank) for i in range(u.ndim))
    v_slice = tuple(slice(None) if i != v_axis else slice(rank) for i in range(v.ndim))
    s_slice = slice(rank)
    return u[u_slice], s[s_slice], v[v_slice]
