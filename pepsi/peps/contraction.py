"""
This module wraps the PEPS contraction routines from https://github.com/HaoTy/PEPS
"""

import numpy as np

import peps


def contract_peps(grid):
    peps_obj = peps.PEPS(grid, insert_pseudo=False, idx_order=[0,3,2,1,4])
    result = peps_obj.contract().match_virtual().reshape(*[2]*grid.shape[0]*grid.shape[1])
    result = np.transpose(result, [i+j*grid.shape[0] for i, j in np.ndindex(*grid.shape)])
    return result

def contract_peps_value(grid):
    peps_obj = peps.PEPS(grid, insert_pseudo=False, idx_order=[0,3,2,1])
    return peps_obj.contract()

def contract_inner(this, that):
    this = peps.PEPS(this, insert_pseudo=False, idx_order=[0,3,2,1,4])
    that = peps.PEPS(that, insert_pseudo=False, idx_order=[0,3,2,1,4])
    return this.inner(that).contract()


def create_env_cache(grid):
    peps_obj = peps.PEPS(grid, insert_pseudo=False, idx_order=[0,3,2,1,4]).norm()
    _up, _down = {}, {}
    for i in range(peps_obj.shape[0]):
        _up[i] = peps_obj[:i].contract_to_MPS() if i != 0 else None
        _down[i] = peps_obj[i+1:].contract_to_MPS() if i != grid.shape[0] - 1 else None
    return _up, _down

def contract_with_env(this, that, env, up_idx, down_idx):
    this = peps.PEPS(this, insert_pseudo=False, idx_order=[0,3,2,1,4])
    that = peps.PEPS(that, insert_pseudo=False, idx_order=[0,3,2,1,4])
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
