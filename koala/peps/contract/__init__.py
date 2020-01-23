"""
This module wraps the PEPS contraction routines from https://github.com/HaoTy/PEPS
"""

import numpy as np

from . import peps


def to_statevector(grid):
    peps_obj = _create_peps(grid)
    result = peps_obj.contract().reshape(*[2]*grid.shape[0]*grid.shape[1])
    result = grid.backend.transpose(result, [i+j*grid.shape[0] for i, j in np.ndindex(*grid.shape)])
    return result

def to_value(grid):
    peps_obj = _create_scalar_peps(grid)
    return peps_obj.contract()

def inner(this, that):
    this = _create_peps(this)
    that = _create_peps(that)
    return this.inner(that).contract()


def get_vertical_local_pair_env(state, pos):
    """Get the env of site at `pos` and the site below it
    Parameters:
    ===========
    state: pepsi.peps.PEPS
        the PEPS to get env from
    pos: Tuple[int, int]
        the position of the upper site
    Returns:
    ========
    output: List[state.backend.tensor]
        6 env tensors in the following order, assuming the sites look like this
        when looking from upside::
            1
            |
        2 - * - 0
            |
        3 - * - 5
            |
            4
        each env tensor looks like this
            3
            |
        0 - * - 1
            |
            2
        where leg 0 connects to previous env tensor, leg 1 connects to next env tensor,
        leg 2 and 3 connect to sites
        e
    """
    env = _create_peps(state).contract_vertical_pair_env(pos)
    return [state.backend.tensor(tsr.match_virtual()) for tsr in env]


def get_horizontal_local_pair_env(state, pos):
    """Get the env of site at `pos` and the site right to it
    Parameters:
    ===========
    state: pepsi.peps.PEPS
        the PEPS to get env from
    pos: Tuple[int, int]
        the position of the left site
    Returns:
    ========
    output: List[state.backend.tensor]
        6 env tensors in the following order, assuming the sites look like this
        when looking from upside:
            0   5
            |   |
        1 - * - * - 4
            |   |
            2   3
        each env tensor looks like this
            3
            |
        0 - * - 1
            |
            2
        where leg 0 connects to previous env tensor, leg 1 connects to next env tensor,
        leg 2 and 3 connect to sites
    """
    env = _create_peps(state).contract_horizontal_pair_env(pos)
    return [state.backend.tensor(tsr.match_virtual()) for tsr in env]


def create_env_cache(grid):
    peps_obj = _create_peps(grid).norm()
    _up, _down = {}, {}
    for i in range(peps_obj.shape[0]):
        _up[i] = peps_obj[:i].contract_to_MPS() if i != 0 else None
        _down[i] = peps_obj[i+1:].contract_to_MPS() if i != grid.shape[0] - 1 else None
    return _up, _down

def inner_with_env(this, that, env, up_idx, down_idx):
    this = _create_peps(this)
    that = _create_peps(that)
    inner = this.inner(that)
    up, down = env[0][up_idx], env[1][down_idx]
    if up is None and down is None:
        peps_obj = inner
    elif up is None:
        peps_obj = inner.concatenate(down)
    elif down is None:
        peps_obj = up.concatenate(inner)
    else:
        peps_obj = up.concatenate(inner).concatenate(down)
    return peps_obj.contract()


def _create_peps(p):
    return peps.PEPS(p.grid, backend=p.backend)

def _create_scalar_peps(p):
    return peps.PEPS(p.grid, backend=p.backend)

def _unwrap(grid):
    newgrid = np.empty_like(grid)
    for idx, tsr in np.ndenumerate(grid):
        newgrid[idx] = tsr.unwrap()
    return newgrid
