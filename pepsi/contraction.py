import peps

def contract_peps(grid):
    nsite = grid.shape[0] * grid.shape[1]
    peps_obj = peps.PEPS(grid, insert_pseudo=False, idx_order=[0,3,2,1,4])
    return peps_obj.contract().match_virtual().reshape(*[2]*nsite)
