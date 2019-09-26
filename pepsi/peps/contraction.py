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
