"""
This module implements utility functions for PEPS operations.
"""
import numpy as np


def svd_merger(subscripts, u, s, v, merge_with='both'):
    str_u, str_v = subscripts.split(',')
    char_s, = set(str_u) - (set(str_u) - set(str_v))
    if merge_with == 'both':
        s = s ** 0.5
    if not merge_with == 'v':
        u = u.backend.einsum(f'{str_u},{char_s}->{str_u}', u, s)
    if not merge_with == 'u':
        v = v.backend.einsum(f'{str_v},{char_s}->{str_v}', v, s)
    return u, v

def vector_reshaper_BMPS(vector, peps_shape):
    return vector.item() if vector.size == 1 else vector.reshape(
        *[int(round(vector.size ** (1 / np.prod(peps_shape))))] * np.prod(peps_shape)
        ).transpose(*[i + j * peps_shape[0] for i, j in np.ndindex(*peps_shape)])
