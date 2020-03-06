"""
This module implements utility functions for PEPS operations.
"""


def svd_splitter(subscripts, u, s, v):
    str_u, str_v = subscripts.split(',')
    char_s, = set(str_u) - (set(str_u) - set(str_v))
    s = s ** 0.5
    u = u.backend.einsum(f'{str_u},{char_s}->{str_u}', u, s)
    v = v.backend.einsum(f'{str_v},{char_s}->{str_v}', v, s)
    return u, v
