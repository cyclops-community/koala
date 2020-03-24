from heisenberg import Heisenberg2D
from experiments.benchmark import Benchmark
from koala import Observable, statevector, peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD
import argparse


def run(args):
    maxrank = args.maxrank
    model = Heisenberg2D(
        args.nrow, args.ncol,
        Jx=[-1.0, -1.0],
        Jy=[-1.0, -1.0],
        Jz=[-1.0, -1.0],
        hx=-1.25, hy=-1.25, hz=-1.25,
    )

    # ground states
    print('exact', model.ground_state_energy()/model.nsite)


    # ITE
    nstep = args.steps
    trotter_step = model.trotter_step(0.01)
    hamiltonian = model.hamiltonian_koala()

    ## statevector
    qstate = statevector.computational_zeros(model.nsite, backend='numpy')
    for i in range(nstep):
        for operator, sites in trotter_step:
            qstate.apply_operator(operator, sites)
        qstate /= qstate.norm()
    print('statevector', qstate.expectation(hamiltonian)/model.nsite)

    ## peps
    update_option = peps.DefaultUpdate(rank=maxrank)
    qstate = peps.computational_zeros(model.nrow, model.ncol, backend=args.backend)
    for i in range(nstep):
        for operator, sites in trotter_step:
            qstate.apply_operator(operator, sites, update_option=update_option)
        qstate.site_normalize()

    include = (Snake, BMPS, ABMPS, TRG, Square)
    exclude = ()
    standard = None
    maxrank = maxrank ** 2

    options = []
    for contract_option in include:
        if contract_option not in exclude:
            if contract_option is Snake:
                options.append(Snake())
            else:
                for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                    options.append(TRG(svd_option, svd_option) if contract_option is TRG else contract_option(svd_option))
    
    for option in options:
        bm = Benchmark(str(option), qstate.backend, standard=standard, path=args.path, 
            reps=1, profile_time=args.profile_time, profile_memory=args.profile_memory, additional_info={})
        bm.add_PEPS_info(qstate)
        bm.add_contract_info(option)
        with bm:
            norm_sq = qstate.inner(qstate, contract_option=option)
            bm.result = qstate.expectation(hamiltonian, contract_option=option)/(norm_sq*model.nsite)

        if isinstance(option, Snake) and not standard:
            standard = bm.result

def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)

    parser.add_argument('-s', '--steps', help='number of steps', type=int, default=300)
    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=2)

    parser.add_argument('-p', '--path', help='path to save the data file', type=str, default=None)
    parser.add_argument('-pt', '--profile-time', dest='profile_time', action='store_const', const='True', 
        default=False, help='enable profiling time')
    parser.add_argument('-pm', '--profile-memory', dest='profile_memory', action='store_const', const='True', 
        default=False, help='enable profiling memory')

    return parser

if __name__ == '__main__':
    parser = build_cli_parser()

    run(parser.parse_args())
