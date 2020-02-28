from itertools import chain
import argparse, time

from koala import statevector, peps, Observable
from koala.peps import sites

import tensorbackends
from tensorbackends.interface import ImplicitRandomizedSVD
import numpy as np
import scipy.linalg as sla


PAULI_X = np.array([[0,1],[1,0]], dtype=complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULI_ZZ = np.einsum('ij,kl->ikjl', PAULI_Z, PAULI_Z)


def one_site_operator(tsr):
    return tsr.reshape(1,1,1,1,2,2)

def horizontal_pair_site_operator(tsr):
    xy, s, uv = tsr.backend.einsvd('xyuv->xys,uvs', tsr)
    s = s ** 0.5
    xy = tsr.backend.einsum('xys,s->()s()()xy', xy, s)
    uv = tsr.backend.einsum('uvs,s->()()()suv', uv, s)
    return xy, uv

def vertical_pair_site_operator(tsr):
    xy, s, uv = tsr.backend.einsvd('xyuv->sxy,suv', tsr)
    s = s ** 0.5
    xy = tsr.backend.einsum('sxy,s->()()s()xy', xy, s)
    uv = tsr.backend.einsum('suv,s->s()()()uv', uv, s)
    return xy, uv

def pepo_identity(nrow, ncol, backend='numpy'):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        grid[i, j] = backend.astensor(np.eye(2,dtype=complex).reshape(1,1,1,1,2,2))
    return peps.PEPS(grid, backend)

def pepo_multiply_operator(pepo, operator, site):
    pepo.grid[site] = sites.contract_z(pepo.grid[site], operator)

def pepo_switch_backend(pepo, backend):
    if pepo.backend.name != 'numpy':
        raise ValueError('can only switch from numpy')
    backend = tensorbackends.get(backend)
    for idx in np.ndindex(*pepo.shape):
        pepo.grid[idx] = backend.astensor(pepo.grid[idx].unwrap())
    pepo.backend = backend
    pepo.using_ctf = backend.name in {'ctf', 'ctfview'}


class Timer:
    def __init__(self, backend, name):
        backend = tensorbackends.get(backend)
        if backend.name in {'ctf', 'ctfview'}:
            import ctf
            self.timer = ctf.timer(name)
        else:
            self.timer = None

    def __enter__(self):
        if self.timer is not None:
            self.timer.start()

    def __exit__(self, type, value, traceback):
        if self.timer is not None:
            self.timer.stop() 


def build_tfi_trottered_step_pepo(J, h, nrows, ncols, tau, backend):
    numpy_backend = tensorbackends.get('numpy')
    one_site = one_site_operator(numpy_backend.astensor(sla.expm(PAULI_X*(h*tau))))
    exp_pauli_zz = numpy_backend.astensor(sla.expm(PAULI_ZZ.reshape(4,4)*(J*tau)).reshape(2,2,2,2))
    h_two_site = horizontal_pair_site_operator(exp_pauli_zz)
    v_two_site = vertical_pair_site_operator(exp_pauli_zz)
    pepo = pepo_identity(nrows, ncols)
    # vertical one site operators
    for i, j in np.ndindex(nrows, ncols):
        pepo_multiply_operator(pepo, one_site, (i,j))
    # horizontal two site operators
    for i, j in np.ndindex(nrows, ncols-1):
        pepo_multiply_operator(pepo, h_two_site[0], (i,j))
        pepo_multiply_operator(pepo, h_two_site[1], (i,j+1))
    # vertical two site operators
    for i, j in np.ndindex(nrows-1, ncols):
        pepo_multiply_operator(pepo, v_two_site[0], (i,j))
        pepo_multiply_operator(pepo, v_two_site[1], (i+1,j))
    pepo_switch_backend(pepo, backend)
    return pepo


def horizontal_links(nrows, ncols):
    # horizontal
    for i, j in np.ndindex(nrows, ncols-1):
        yield (i, j), (i, j+1)

def vertical_links(nrows, ncols):
    # vertical
    for i, j in np.ndindex(nrows-1, ncols):
        yield (i, j), (i+1, j)


def reduce_bond_dimensions(qstate, maxrank):
    for a, b in horizontal_links(qstate.nrow, qstate.ncol):
        qstate.grid[a], qstate.grid[b] = sites.reduce_y(qstate[a], qstate[b], option=ImplicitRandomizedSVD(maxrank))
    for a, b in vertical_links(qstate.nrow, qstate.ncol):
        qstate.grid[a], qstate.grid[b] = sites.reduce_x(qstate[a], qstate[b], option=ImplicitRandomizedSVD(maxrank))


def run_peps(pepo, steps, normfreq, backend, threshold, maxrank, randomized_svd):
    using_ctf = backend in {'ctf', 'ctfview'}
    if using_ctf:
        import ctf
        timer_epoch = ctf.timer_epoch('run_peps')
        timer_epoch.begin()
        ctf.initialize_flops_counter()
    qstate = peps.random(pepo.nrow, pepo.ncol, maxrank, backend=backend)
    if using_ctf:
        ctf.initialize_flops_counter()
    for i in range(steps):
        with Timer(qstate.backend, 'apply_pepo'):
            qstate = pepo.apply(qstate)
        with Timer(qstate.backend, 'reduce_bond_dimensions'):
            reduce_bond_dimensions(qstate, maxrank)
        if (i+1) % normfreq == 0:
            qstate.site_normalize()
    if using_ctf:
        timer_epoch.end()
        flops = ctf.get_estimated_flops()
    else:
        flops = None
    return qstate, flops


def main(args):
    with Timer(args.backend, 'build_pepo'):
        pepo = build_tfi_trottered_step_pepo(args.coupling, args.field, args.nrow, args.ncol, args.tau, args.backend)

    t = time.time()
    peps_qstate, flops = run_peps(pepo, args.steps, args.normfreq, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank, randomized_svd=args.randomized_svd)
    peps_ite_time = time.time() - t

    backend = tensorbackends.get(args.backend)

    if backend.rank == 0:
        print('tfi.nrow', args.nrow)
        print('tfi.ncol', args.ncol)
        print('tfi.field', args.field)
        print('tfi.coupling', args.coupling)

        print('ite.steps', args.steps)
        print('ite.tau', args.tau)
        print('ite.normfreq', args.normfreq)

        print('backend.name', args.backend)
        print('backend.nproc', backend.nproc)

        print('peps.maxrank', args.maxrank)

        print('result.peps_ite_time', peps_ite_time)
        print('result.peps_ite_flops', flops)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)

    parser.add_argument('-cp', '--coupling', help='coupling value of TFI', type=float, default=1.0)
    parser.add_argument('-f', '--field', help='field value of TFI', type=float, default=0.2)

    parser.add_argument('-s', '--steps', help='ITE steps', type=int, default=100)
    parser.add_argument('-tau', help='ITE trotter size', type=float, default=0.01)
    parser.add_argument('-nf', '--normfreq', help='ITE normalization frequency', type=int, default=10)

    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-th', '--threshold', help='the threshold in trucated SVD when applying gates', type=float, default=1e-5)
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=1)
    parser.add_argument('-rsvd', '--randomized_svd', help='use randomized SVD when applying gates', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())