import argparse, time

from koala import peps

import tensorbackends
from tensorbackends.interface import ImplicitRandomizedSVD
import numpy as np


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

class TimerEpoch:
    def __init__(self, backend, name):
        backend = tensorbackends.get(backend)
        if backend.name in {'ctf', 'ctfview'}:
            import ctf
            self.timer_epoch = ctf.timer_epoch(name)
        else:
            self.timer_epoch = None

    def __enter__(self):
        if self.timer_epoch is not None:
            self.timer_epoch.begin()

    def __exit__(self, type, value, traceback):
        if self.timer_epoch is not None:
            self.timer_epoch.end()


def random_scalar_peps(nrow, ncol, rank, backend):
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        shape = (
            rank if i > 0 else 1,
            rank if j < ncol - 1 else 1,
            rank if i < nrow - 1 else 1,
            rank if j > 0 else 1,
            1, 1,
        )
        grid[i, j] = backend.random.uniform(-1,1,shape) + 1j * backend.random.uniform(-1,1,shape)
    return peps.PEPS(grid, backend)


def run_peps(nrow, ncol, maxrank, backend):
    qstate = random_scalar_peps(nrow, ncol, maxrank, backend)
    return qstate.contract(option=peps.BMPS(svd_option=ImplicitRandomizedSVD(maxrank)))


def main(args):
    t = time.process_time()
    with TimerEpoch(args.backend, 'PEPS_contract'):
        peps_qstate = run_peps(args.nrow, args.ncol, args.maxrank, args.backend)
    peps_contract_time = time.process_time() - t

    if args.backend in {'ctf', 'ctfview'}:
        import ctf
        peps_contract_flops = ctf.get_estimated_flops()
    else:
        peps_contract_flops = None

    backend = tensorbackends.get(args.backend)
    if backend.rank == 0:
        print('tfi.nrow', args.nrow)
        print('tfi.ncol', args.ncol)

        print('backend.name', args.backend)
        print('backend.nproc', backend.nproc)

        print('peps.maxrank', args.maxrank)

        print('result.peps_contract_time', peps_contract_time)
        print('result.peps_contract_flops', peps_contract_flops)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)
    
    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=2)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())
