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


def split_pair_site_operator(tsr):
    xy, s, uv = tsr.backend.einsvd('xuyv->xys,uvs', tsr)
    s = s ** 0.5
    xys = tsr.backend.einsum('xys,s->xys', xy, s)
    uvs = tsr.backend.einsum('uvs,s->uvs', uv, s)
    return xys, uvs

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

class TFI:
    def __init__(self, J, h, nrows, ncols, tau=0.01, backend='numpy'):
        self.backend = tensorbackends.get(backend)
        self.J = J
        self.h = h
        self.nrows = nrows
        self.ncols = ncols
        self.tau = tau

        numpy_backend = tensorbackends.get('numpy')
        backend = tensorbackends.get(backend)
        one_site = numpy_backend.astensor(sla.expm(PAULI_X*(h*tau)))
        exp_pauli_zz = numpy_backend.astensor(sla.expm(PAULI_ZZ.reshape(4,4)*(J*tau)).reshape(2,2,2,2))
        two_site = split_pair_site_operator(exp_pauli_zz)
        self.one_site = backend.astensor(one_site.unwrap())
        self.two_site = tuple(backend.astensor(tsr.unwrap()) for tsr in two_site)

    def single(self):
        for i, j in np.ndindex(self.nrows, self.ncols):
            yield (i, j)

    def double_horizontal(self):
        # horizontal double gate positions
        for i, j in np.ndindex(self.nrows, self.ncols-1):
            yield (i, j), (i, j+1)

    def double_vertical(self):
        # vertical double gate positions
        for i, j in np.ndindex(self.nrows-1, self.ncols):
            yield (i, j), (i+1, j)

    def apply_trottered_step(self, qstate, maxrank):
        with Timer(qstate.backend, 'apply_trottered_step_single_contraction'):
            for idx in self.single():
                qstate.grid[idx] = qstate.backend.einsum('ijklxp,xy->ijklyp', qstate.grid[idx], self.one_site)
        for a, b in self.double_horizontal():
            apply_operator(qstate, a, b, *self.two_site, maxrank, orientation='h')
        for a, b in self.double_vertical():
            apply_operator(qstate, a, b, *self.two_site, maxrank, orientation='v')


def apply_operator(qstate, x_pos, y_pos, x_operator, y_operator, maxrank, orientation):
    x, y = qstate.grid[x_pos], qstate.grid[y_pos]

    if orientation == 'v':
        apply_on_x = 'abcdxp,xuA->(abdup)(cA)'
        apply_on_y = 'cfghyp,yvA->(cA)(fghvp)'
        m_axes = [0, 1, 3, 4, 5]
        n_axes = [1, 2, 3, 4, 5]
        extract_x = 'abdupc,c->abcdup'
        extract_y = 'cfghvp,c->cfghvp'
    elif orientation == 'h':
        apply_on_x = 'abcdxp,xuA->(acdup)(bA)'
        apply_on_y = 'efgbyp,yvA->(bA)(efgvp)'
        m_axes = [0, 2, 3, 4, 5]
        n_axes = [0, 1, 2, 4, 5]
        extract_x = 'acdupb,b->abcdup'
        extract_y = 'befgvp,b->efgbvp'
    else:
        assert False

    m_shape = [x.shape[d] for d in m_axes]
    n_shape = [y.shape[d] for d in n_axes]

    with Timer(qstate.backend, 'apply_trottered_step_double_contraction'):
        x_mat = qstate.backend.einsum(apply_on_x, x, x_operator)
        y_mat = qstate.backend.einsum(apply_on_y, y, y_operator)

    with Timer(qstate.backend, 'reduce_bond_dimensions'):
        u, s, vh = randomized_svd(qstate.backend, x_mat, y_mat, maxrank, niter=1, oversamp=5)
        with Timer(qstate.backend, 'reduce_bond_dimensions_multiply_s'):
            s = s ** 0.5
            x = qstate.backend.einsum(extract_x, u.reshape(*m_shape, -1), s)
            y = qstate.backend.einsum(extract_y, vh.reshape(-1, *n_shape), s)

    qstate.grid[x_pos] = x
    qstate.grid[y_pos] = y


def randomized_svd(backend, a, b, rank, niter=1, oversamp=5):
    using_ctf = backend.name in {'ctf', 'ctfview'}
    if using_ctf:
        import ctf
        timer = ctf.timer('randomized_svd')
        timer.start()
    m, n = a.shape[0], b.shape[1]
    assert a.shape[1] == b.shape[0]
    r = min(rank + oversamp, m, n, a.shape[1])
    # find subspace
    q = backend.random.uniform(low=-1.0, high=1.0, size=(n, r)).astype(complex)
    a_H, b_H = a.H, b.H
    for i in range(niter):
        q = b_H @ (a_H @ (a @ (b @ q)))
        q, _ = backend.qr(q)
    q = a @ (b @ q)
    q, _ = backend.qr(q)
    # svd
    ab_sub = (q.H @ a) @ b
    u_sub, s, vh = backend.svd(ab_sub)
    u = q @ u_sub
    if rank < r:
        u, s, vh = u[:,:rank], s[:rank], vh[:rank,:]
    if using_ctf:
        timer.stop()
    return u, s, vh


def run_peps(tfi, nrow, ncol, steps, normfreq, backend, threshold, maxrank, randomized_svd):
    using_ctf = backend in {'ctf', 'ctfview'}
    if using_ctf:
        import ctf
        timer_epoch = ctf.timer_epoch('run_peps')
        timer_epoch.begin()
        ctf.initialize_flops_counter()
    qstate = peps.random(nrow, ncol, maxrank, backend=backend)
    if using_ctf:
        ctf.initialize_flops_counter()
    for i in range(steps):
        with Timer(qstate.backend, 'apply_trottered_step'):
            tfi.apply_trottered_step(qstate, maxrank)
        if (i+1) % normfreq == 0:
            qstate.site_normalize()
    if using_ctf:
        timer_epoch.end()
        flops = ctf.get_estimated_flops()
    else:
        flops = None
    return qstate, flops


def main(args):
    with Timer(args.backend, 'build_operators'):
        tfi = TFI(args.coupling, args.field, args.nrow, args.ncol, args.tau, args.backend)

    t = time.time()
    peps_qstate, flops = run_peps(tfi, args.nrow, args.ncol, args.steps, args.normfreq, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank, randomized_svd=args.randomized_svd)
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