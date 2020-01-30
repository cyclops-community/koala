from string import ascii_letters as chars

from .contract import get_horizontal_local_pair_env, get_vertical_local_pair_env

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
    u, s, v = truncate(state.backend, u, s, v, link[0], link[1], threshold=threshold, maxrank=maxrank)
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


def apply_full_update(state, operator, pos1, pos2, rank, epsilon=1e-5, reg=1e-7):
    """Apply full update to two sites

    Parameters
    ----------
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
    env_str = "01aA,12bB,23cC,34dD,45eE,50fF"

    if pos2[0] == pos1[0]:
        assert pos2[1] == pos1[1] + 1
        env = get_horizontal_local_pair_env(state, pos1)
        sites_w_operator = backend.einsum("abczl,fzdem,lmop->abcdefop", site1, site2, operator)
        rhs = backend.einsum(f"{env_str},abcdefop->opABCDEF", *env, sites_w_operator)
        site1, s, site2 = backend.einsvd("abcdefop->abclo,fldep", sites_w_operator)
        site1, s, site2 = truncate(backend, site1, s, site2, 3, 1, threshold=None, maxrank=rank)
        s **= 0.5
        site1 = backend.einsum('abclo,l->abclo', site1, s)
        site2 = backend.einsum('fldep,l->fldep', site2, s)
        rel_residual = 1.
        while rel_residual > epsilon:
            site1_new, site2_new = low_rank_update_step(backend, env, rhs, site1, site2, mode='horizontal', reg=reg)
            # check the residual
            rel_residual = backend.norm(site1_new - site1) / backend.norm(site1) + backend.norm(site2_new - site2) / backend.norm(site2)
            site1, site2 = site1_new, site2_new
    else:
        assert pos2[0] == pos1[0] + 1 and pos2[1] == pos1[1]
        env = get_vertical_local_pair_env(state, pos1)
        sites_w_operator = backend.einsum("bczal,zdefm,lmop->abcdefop", site1, site2, operator)
        rhs = backend.einsum(f"{env_str},abcdefop->opABCDEF", *env, sites_w_operator)
        site1, s, site2 = backend.einsvd("abcdefop->bclao,ldefp", sites_w_operator)
        site1, s, site2 = truncate(backend, site1, s, site2, 2, 0, threshold=None, maxrank=rank)
        s **= 0.5
        site1 = backend.einsum('bclao,l->bclao', site1, s)
        site2 = backend.einsum('ldefp,l->ldefp', site2, s)
        rel_residual = 1.
        while rel_residual > epsilon:
            site1_new, site2_new = low_rank_update_step(backend, env, rhs, site1, site2, mode='vertical', reg=reg)
            # check the residual
            rel_residual = backend.norm(site1_new - site1) / backend.norm(site1) + backend.norm(site2_new - site2) / backend.norm(site2)
            site1, site2 = site1_new, site2_new
    state.grid[pos1], state.grid[pos2] = site1, site2
    return state


def low_rank_update_step(backend, env, rhs, site1, site2, mode='horizontal', reg=1e-7):
    env_str = "01aA,12bB,23cC,34dD,45eE,50fF"
    if mode == 'vertical':
        site_strs = ['ldefp', 'LDEFp', 'bclao', 'BCLAo']
    elif mode == 'horizontal':
        site_strs = ['fldep', 'FLDEp', 'abclo', 'ABCLo']
    # update site1
    env_site = backend.einsum(f"{env_str},{site_strs[0]},{site_strs[1]}->abclABCL", *env, site2, site2)
    rhs_site = backend.einsum(f"opABCDEF,{site_strs[1]}->oABCL", rhs, site2)
    length = env_site.shape[0] * env_site.shape[1] * env_site.shape[2] * env_site.shape[3]
    env_tensor_shape = env_site.shape
    env_site = env_site.reshape((length, length))
    # add regularization
    env_site_reg = env_site + reg * backend.identity(env_site.shape[0])
    inv_env_site = backend.inv(env_site_reg).reshape(*env_tensor_shape)
    site1_new = backend.einsum(f"oABCL,ABCLabcl->{site_strs[2]}", rhs_site, inv_env_site)
    # update site2
    env_site = backend.einsum(f"{env_str},{site_strs[2]},{site_strs[3]}->deflDEFL", *env, site1, site1)
    rhs_site = backend.einsum(f"opABCDEF,{site_strs[3]}->pDEFL", rhs, site1)
    length = env_site.shape[0] * env_site.shape[1] * env_site.shape[2] * env_site.shape[3]
    env_tensor_shape = env_site.shape
    env_site = env_site.reshape((length, length))
    # add regularization
    env_site_reg = env_site + reg * backend.identity(env_site.shape[0])
    inv_env_site = backend.inv(env_site_reg).reshape(*env_tensor_shape)
    site2_new = backend.einsum(f"pDEFL,DEFLdefl->{site_strs[0]}", rhs_site, inv_env_site)
    return site1_new, site2_new


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
