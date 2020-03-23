from tfi_ite_peps import TraversalFieldIsing, build_cli_parser, run_peps
from benchmark_contract import benchmark_expectation, save_data
from koala import peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD


def update_save_data(data, args, rank):
    data['coupling'] = args.coupling
    data['field'] = args.field
    data['tau'] = args.tau
    data['steps'] = args.steps
    data['normfreq'] = args.normfreq
    if args.path and rank == 0:
        save_data(data, args.path)
    return data


def run(args):
    tfi = TraversalFieldIsing(args.coupling, args.field, args.nrow, args.ncol, args.tau, args.backend)
    qstate = run_peps(tfi, args.steps, args.normfreq, backend=args.backend, maxrank=args.maxrank)
    standard = update_save_data(benchmark_expectation(qstate, tfi.observable, contract_option=Snake(), 
        standard=None, path=args.path, reps=1, profile=args.profile), args, qstate.backend.rank)
    maxrank = args.maxrank # ** 2
    for contract_option in contract_options:
        if contract_option not in (Snake, ):
            for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                update_save_data(benchmark_expectation(qstate, tfi.observable, 
                    TRG(svd_option, svd_option) if contract_option is TRG else contract_option(svd_option), 
                    standard=standard['result'], path=args.path, reps=1, profile=args.profile), args, qstate.backend.rank)



if __name__ == '__main__':
    parser = build_cli_parser()
    parser.add_argument('-p', '--path', help='path to save the datafile', type=str, default=None)
    parser.add_argument('--prof', '--profile', dest='profile', action='store_const', const='True', 
        default=False, help='enable profiling')

    run(parser.parse_args())
