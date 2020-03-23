from itertools import chain
import argparse, time

from koala import statevector, peps, Observable

import tensorbackends
import numpy as np
import scipy.linalg as sla


PAULI_X = np.array([[0,1],[1,0]], dtype=complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULI_ZZ = np.einsum('ij,kl->ikjl', PAULI_Z, PAULI_Z)


class RandomIsing:
    def __init__(self, J, h, random_ratio, seed, nrows, ncols,  tau=0.01, backend='numpy'):
        self.backend = tensorbackends.get(backend)
        self.J = J
        self.h = h
        self.nrows = nrows
        self.ncols = ncols
        self.seed = seed
        self.random_ratio = random_ratio
        self.rnd = np.random.RandomState(seed)
        self.fields = {pos: self.rnd.uniform(-h, h) if self.rnd.random() < random_ratio else h for pos in self.single_positions()}
        self.couplings = {pos: self.rnd.uniform(-J, J) if self.rnd.random() < random_ratio else J for pos in self.double_positions()}
        self.tau = tau
        self.trottered_single_gate = [
            self.backend.astensor(sla.expm(PAULI_X*(self.fields[pos]*tau)))
            for pos in self.single_positions()
        ]
        self.trottered_double_gate = [
            self.backend.astensor(sla.expm(PAULI_ZZ.reshape(4,4)*(self.couplings[pos]*tau)).reshape(2,2,2,2))
            for pos in self.double_positions()
        ]
        self.observable = Observable.sum(chain(
            (Observable.operator(self.backend.astensor(PAULI_X*(-self.fields[pos])), pos) for pos in self.single_positions()),
            (Observable.operator(self.backend.astensor(PAULI_ZZ*(-self.couplings[pos])), pos) for pos in self.double_positions()),
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
        yield from zip(self.trottered_single_gate, self.single_positions())
        yield from zip(self.trottered_double_gate, self.double_positions())


def run_statevector(ri, steps, normfreq, backend):
    qstate = statevector.computational_zeros(ri.nrows*ri.ncols, backend=backend)
    for i in range(steps):
        for operator, sites in ri.trotter_steps():
            qstate.apply_operator(operator, sites)
        if i % normfreq == 0:
            qstate /= qstate.norm()
    qstate /= qstate.norm()
    return qstate


def run_peps(ri, steps, normfreq, backend, threshold, maxrank, randomized_svd):
    qstate = peps.computational_zeros(ri.nrows, ri.ncols, backend=backend)
    for i in range(steps):
        for operator, sites in ri.trotter_steps():
            qstate.apply_operator(operator, sites, update_option=peps.DefaultUpdate(maxrank))
        if i % normfreq == 0:
            qstate.site_normalize()
    qstate /= qstate.norm()
    return qstate


def main(args):
    ri = RandomIsing(args.coupling, args.field, args.random_ratio, args.random_seed, args.nrow, args.ncol, args.tau, args.backend)

    t = time.process_time()
    statevector_qstate = run_statevector(ri, args.steps, args.normfreq, backend=args.backend)
    statevector_ite_time = time.process_time() - t

    t = time.process_time()
    statevector_energy = statevector_qstate.expectation(ri.observable)
    statevector_expectiation_time = time.process_time() - t

    t = time.process_time()
    peps_qstate = run_peps(ri, args.steps, args.normfreq, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank, randomized_svd=args.randomized_svd)
    peps_ite_time = time.process_time() - t

    t = time.process_time()
    peps_energy = peps_qstate.expectation(ri.observable, use_cache=True)
    peps_expectation_time = time.process_time() - t

    backend = tensorbackends.get(args.backend)

    if backend.rank == 0:
        print('ri.nrow', args.nrow)
        print('ri.ncol', args.ncol)
        print('ri.field', args.field)
        print('ri.coupling', args.coupling)
        print('ri.random_ratio', args.random_ratio)
        print('ri.random_seed', args.random_ratio)

        print('ite.steps', args.steps)
        print('ite.tau', args.tau)
        print('ite.normfreq', args.normfreq)

        print('backend.name', args.backend)
        print('backend.nproc', backend.nproc)

        print('peps.threshold', args.threshold)
        print('peps.maxrank', args.maxrank)

        print('result.statevector_energy', statevector_energy)
        print('result.peps_energy', peps_energy)

        print('result.statevector_ite_time', statevector_ite_time)
        print('result.peps_ite_time', peps_ite_time)

        print('result.statevector_expectiation_time', statevector_expectiation_time)
        print('result.peps_expectation_time', peps_expectation_time)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)

    parser.add_argument('-cp', '--coupling', help='coupling value of Ising', type=float, default=1.0)
    parser.add_argument('-f', '--field', help='field value of Ising', type=float, default=0.2)
    parser.add_argument('-rr', '--random_ratio', help='random ratio of choosing couplings/fields randomly', type=float, default=0.5)
    parser.add_argument('-rs', '--random_seed', help='random seed to sample the model', type=int, default=None)

    parser.add_argument('-s', '--steps', help='ITE steps', type=int, default=100)
    parser.add_argument('-tau', help='ITE trotter size', type=float, default=0.01)
    parser.add_argument('-nf', '--normfreq', help='ITE normalization frequency', type=int, default=10)

    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-th', '--threshold', help='the threshold in trucated SVD when applying gates', type=float, default=1e-5)
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=None)
    parser.add_argument('-rsvd', '--randomized_svd', help='use randomized SVD when applying gates', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
