"""
This module implements operations on PEPS sites.

Here are the convetions for names of directions and order of axes.

Direction names:        |   Axes order:
          z             |            4 0
          |/            |            |/
       -- O -- y        |       3 -- O -- 1
         /|             |           /|
        x               |          2 5
"""

import numpy as np


def contract_x(a, b):
    return a.backend.einsum('abidpq,iBcDPQ->a(bB)c(dD)(pP)(qQ)', a, b)

def contract_y(a, b):
    return a.backend.einsum('aicdpq,AbCiPQ->(aA)b(cC)d(pP)(qQ)', a, b)

def contract_z(a, b):
    return a.backend.einsum('abcdpi,ABCDiq->(aA)(bB)(cC)(dD)pq', a, b)


def reduce_x(a, b):
    return a.backend.einsumsvd('abidpq,iBcDPQ->abIdpq,IBcDPQ', a, b)

def reduce_y(a, b):
    return a.backend.einsumsvd('aicdpq,AbCiPQ->aIcdpq,AbCIPQ', a, b)

def reduce_z(a, b):
    return a.backend.einsumsvd('abcdpi,ABCDiq->abcdpI,ABCDIq', a, b)


def rotate_x(a, n=1):
    p = np.roll([4, 1, 5, 3], n)
    return a.transpose(0, p[1], 2, p[3], p[0], p[2])

def rotate_y(a, n=1):
    p = np.roll([4, 0, 5, 2], n)
    return a.transpose(p[2], 1, p[4], 3, p[0], p[3])

def rotate_z(a, n=1):
    p = np.roll([0, 1, 2, 3], n)
    return a.transpose(p[0], p[1], p[2], p[3], 4, 5)


def flip_x(a):
    return a.transpose(2, 1, 0, 3, 4, 5)

def flip_y(a):
    return a.transpose(0, 3, 2, 1, 4, 5)

def flip_z(a):
    return a.transpose(0, 1, 2, 3, 5, 4)
