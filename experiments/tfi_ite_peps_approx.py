from itertools import chain
import argparse, time

from koala import statevector, peps, Observable

import tensorbackends
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD
import numpy as np
import scipy.linalg as sla


PAULI_X = np.array([[0,1],[1,0]], dtype=complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULI_ZZ = np.einsum('ij,kl->ikjl', PAULI_Z, PAULI_Z)


class TraversalFieldIsing:
    def __init__(self, J, h, nrows, ncols, tau=0.01, backend='numpy'):
        self.backend = tensorbackends.get(backend)
        self.J = J
        self.h = h
        self.nrows = nrows
        self.ncols = ncols
        self.tau = tau
        self.hamiltionian_single = self.backend.astensor(PAULI_X*(-h))
        self.hamiltionian_double = self.backend.astensor(PAULI_ZZ*(-J))
        self.trottered_single_gate = self.backend.astensor(sla.expm(PAULI_X*(h*tau)))
        self.trottered_double_gate = self.backend.astensor(sla.expm(PAULI_ZZ.reshape(4,4)*(J*tau)).reshape(2,2,2,2))
        self.observable = Observable.sum(chain(
            (Observable.operator(self.hamiltionian_single, pos) for pos in self.single_positions()),
            (Observable.operator(self.hamiltionian_double, pos) for pos in self.double_positions()),
        ))

    def single_positions(self):
        for i in range(self.nrows * self.ncols):
            yield (i,)

    def double_positions(self):
        # horizontal double gate positions
        for i, j in np.ndindex(self.nrows, self.ncols-1):
            yield (i*self.ncols+j, i*self.ncols+j+1)
        # vertical double gate positions
        for i, j in np.ndindex(self.nrows-1, self.ncols):
            yield (i*self.ncols+j, (i+1)*self.ncols+j)
    
    def trotter_steps(self):
        for pos in self.single_positions():
            yield self.trottered_single_gate, pos
        for pos in self.double_positions():
            yield self.trottered_double_gate, pos


def run_peps(tfi, steps, normfreq, backend, threshold, maxrank, randomized_svd):
    qstate = peps.computational_zeros(tfi.nrows, tfi.ncols, backend=backend)
    for i in range(steps):
        for operator, sites in tfi.trotter_steps():
            qstate.apply_operator(operator, sites, threshold=threshold, maxrank=maxrank, randomized_svd=randomized_svd)
        if i % normfreq == 0:
            qstate.site_normalize()
    qstate /= qstate.norm()
    return qstate


def main(args):
    tfi = TraversalFieldIsing(args.coupling, args.field, args.nrow, args.ncol, args.tau, args.backend)

    t = time.process_time()
    peps_qstate = run_peps(tfi, args.steps, args.normfreq, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank, randomized_svd=args.randomized_svd)
    peps_ite_time = time.process_time() - t

    t = time.process_time()
    peps_energy_svd_approx = peps_qstate.expectation(tfi.observable, use_cache=True, contract_option=peps.BMPS(ReducedSVD(rank=args.maxrank**2)))
    peps_svd_approx_expectation_time = time.process_time() - t

    t = time.process_time()
    peps_energy_rsvd_approx = peps_qstate.expectation(tfi.observable, use_cache=True, contract_option=peps.BMPS(RandomizedSVD(rank=args.maxrank**2)))
    peps_rsvd_approx_expectation_time = time.process_time() - t

    t = time.process_time()
    peps_energy_irsvd_approx = peps_qstate.expectation(tfi.observable, use_cache=True, contract_option=peps.BMPS(ImplicitRandomizedSVD(rank=args.maxrank**2)))
    peps_irsvd_approx_expectation_time = time.process_time() - t

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

        print('peps.threshold', args.threshold)
        print('peps.maxrank', args.maxrank)

        print('result.peps_energy_svd_approx', peps_energy_svd_approx)
        print('result.peps_energy_rsvd_approx', peps_energy_rsvd_approx)
        print('result.peps_energy_irsvd_approx', peps_energy_irsvd_approx)

        print('result.peps_ite_time', peps_ite_time)

        print('result.peps_svd_approx_expectation_time', peps_svd_approx_expectation_time)
        print('result.peps_rsvd_approx_expectation_time', peps_rsvd_approx_expectation_time)
        print('result.peps_irsvd_approx_expectation_time', peps_irsvd_approx_expectation_time)


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
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=2)
    parser.add_argument('-rsvd', '--randomized_svd', help='use randomized SVD when applying gates', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())