from rqc import generate, build_cli_parser, run_peps
from benchmark_contract import benchmark_norm, save_data
from koala import peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD


def update_save_data(data, args):
    data['nlayer'] = args.nlayer
    data['seed'] = args.seed
    if args.path:
        save_data(data, args.path)
    return data


def run(args):
    qstate = run_peps(generate(args.nrow, args.ncol, args.nlayer, args.seed), args.maxrank, args.backend)
    standard = update_save_data(benchmark_norm(
        qstate, contract_option=Snake(), path=args.path, reps=2, profile=args.profile), args)
    maxrank = args.maxrank ** 2
    for contract_option in contract_options:
        if contract_option is not Snake:
            for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                update_save_data(benchmark_norm(qstate, 
                    TRG(svd_option, svd_option) if contract_option is TRG else contract_option(svd_option), 
                    standard=standard['result'], path=args.path, reps=2, profile=args.profile), args)



if __name__ == '__main__':
    parser = build_cli_parser()
    parser.add_argument('-p', '--path', help='path to save the datafile', type=str, default=None)
    parser.add_argument('--prof', '--profile', dest='profile', action='store_const', const='True', 
        default=False, help='enable profiling')

    run(parser.parse_args())
