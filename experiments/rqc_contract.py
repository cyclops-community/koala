from rqc import generate, build_cli_parser, run_peps
from benchmark_contract import benchmark_norm
from koala import peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD


def run(args):
    qstate = run_peps(generate(args.nrow, args.ncol, args.nlayer, args.seed), args.maxrank, args.backend)
    standard = benchmark_norm(qstate, contract_option=Snake(), path=args.path, reps=1, profile=args.profile)
    maxrank = args.maxrank ** 2
    for contract_option in contract_options:
        if contract_option is not Snake:
            for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                benchmark_norm(qstate, contract_option=contract_option(svd_option), standard=standard['result'], path=args.path, reps=1, profile=args.profile)



if __name__ == '__main__':
    parser = build_cli_parser()
    parser.add_argument('-p', '--path', help='path to save the datafile', type=str, default=None)
    parser.add_argument('--prof', '--profile', dest='profile', action='store_const', const='True', 
        default=False, help='enable profiling')

    run(parser.parse_args())
