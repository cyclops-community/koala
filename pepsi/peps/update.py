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


def apply_full_update(state, operator, pos1, pos2, rank, epsilon=1e-5):
    """Apply full update to two sites

    Parameters:
    ===========
    state: pepsi.peps.PEPS
        the PEPS to modify

    operator: state.backend.tensor
        operator tensor of shape (pos1_input, pos2_input, pos1_output, pos2_output)

    pos1: Tuple[int, int]
        the position of the first site

    pos2: Tuple[int, int]
        the position of the second site (adjacent to `pos1`)

    rank: int
        new bond dim between the updated sites

    epsilon: double
        the criteria to stop the low rank iterations
    """
    site1, site2 = state[pos1], state[pos2]
    backend = state.backend
    env_str = "0aA1,1bB2,2cC3,3dD4,4eE5,5fF0"

    if pos2[0] == pos1[0]:
        assert pos2[1] == pos1[1] + 1
        env = get_vertical_local_pair_env(state, pos1)
        sites_w_operator = backend.einsum("bczal,zdefm,lmop->abcdefop", site1,
                                          site2, operator)
        rhs = backend.einsum(f"{env_str},abcdefop->opABCDEF", *env,
                             sites_w_operator)
        site1, site2 = backend.einsvd("abcdefop->bclao,ldefp",
                                      sites_w_operator)
        residual = 1.
        while residual > epsilon:
            site1_new, site2_new = low_rank_update_step(backend, env, rhs, site1, site2, mode='vertical')
            # check the residual
            residual = backend.norm(site1_new -
                                    site1) + backend.norm(site2_new - site2)
            site1, site2 = site1_new, site2_new
    else:
        assert pos2[0] == pos1[0] + 1 and pos2[1] == pos1[0]
        env = get_horizontal_local_pair_env(state, pos1)
        sites_w_operator = backend.einsum("abczl,fzdem,lmop->abcdefop", site1,
                                          site2, operator)
        rhs = backend.einsum(f"{env_str},abcdefop->opABCDEF", *env,
                             sites_w_operator)
        site1, site2 = backend.einsvd("abcdefop->abclo,fldep",
                                      sites_w_operator)
        residual = 1.
        while residual > epsilon:
            site1_new, site2_new = low_rank_update_step(backend, env, rhs, site1, site2, mode='horizontal')
            # check the residual
            residual = backend.norm(site1_new -
                                    site1) + backend.norm(site2_new - site2)
            site1, site2 = site1_new, site2_new
    state[pos1], state[pos2] = site1, site2
    return state


def low_rank_update_step(backend, env, rhs, site1, site2, mode='horizontal'):
    env_str = "0aA1,1bB2,2cC3,3dD4,4eE5,5fF0"
    if mode == 'vertical':
        site_strs = ['ldefp', 'LDEFp', 'bclao', 'BCLAo']
    elif mode == 'horizontal':
        site_strs = ['fldep', 'FLDEp', 'abclo', 'ABCLo']
    # update site1
    env_site = backend.einsum(
        f"{env_str},{site_strs[0]},{site_strs[1]}->abclABCL", *env, site2,
        site2)
    rhs_site = backend.einsum(f"opABCDEF,{site_strs[1]}->oABCL", rhs, site2)
    length = env_site.shape[0] * env_site.shape[1] * env_site.shape[
        2] * env_site.shape[3]
    env_tensor_shape = env_site.shape
    env_site = env_site.reshape((length, length))
    inv_env_site = inv(backend, env_site).reshape(*env_tensor_shape)
    site1_new = backend.einsum(f"oABCL,ABCLabcl->{site_strs[2]}", rhs_site,
                               inv_env_site)
    # update site2
    env_site = backend.einsum(
        f"abcdefABCDEF,{site_strs[2]},{site_strs[3]}->deflDEFL", env, site2,
        site2)
    rhs_site = backend.einsum(f"opABCDEF,{site_strs[3]}->pDEFL", rhs, site2)
    length = env_site.shape[0] * env_site.shape[1] * env_site.shape[
        2] * env_site.shape[3]
    env_tensor_shape = env_site.shape
    env_site = env_site.reshape((length, length))
    inv_env_site = inv(backend, env_site).reshape(*env_tensor_shape)
    site2_new = backend.einsum(f"pDEFL,DEFLdefl->{site_strs[0]}", rhs_site,
                               inv_env_site)
    return site1_new, site2_new


def inv(backend, matrix):
    if backend.name == 'ctf':
        U, s, V = backend.svd(matrix)
        return backend.dot(
            backend.transpose(V),
            backend.dot(backend.diag(s**-1), backend.transpose(U)))
    elif backend.name == 'numpy':
        return backend.inv(matrix)


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
