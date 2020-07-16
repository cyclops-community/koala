"""
This module implements utility functions for PEPS operations.
"""


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
