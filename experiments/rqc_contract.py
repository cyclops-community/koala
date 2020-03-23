from experiments.rqc import generate, build_cli_parser, run_peps
from experiments.benchmark import Benchmark
from koala import peps
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD



def run(args):
    include = (Snake, BMPS, ABMPS, TRG, Square)
    exclude = ()
    qstate = run_peps(generate(args.nrow, args.ncol, args.nlayer, args.seed), args.maxrank, args.backend)
    maxrank = args.maxrank ** 2
    standard = None

    for contract_option in include:
        if contract_option not in exclude:
            if contract_option is Snake:
                option = Snake()
            else:
                for svd_option in (ReducedSVD(maxrank), RandomizedSVD(maxrank), ImplicitRandomizedSVD(maxrank)):
                    option = TRG(None, svd_option) if contract_option is TRG else contract_option(svd_option)
            
            bm = Benchmark(str(option), qstate.backend, standard=standard, path=args.path, 
                reps=1, profile=args.profile, additional_info={'nlayer': args.nlayer, 'seed': args.seed})
            bm.add_PEPS_info(qstate)
            bm.add_contract_info(option)
            with bm:
                bm.result = qstate.norm(contract_option=option)
            
            if contract_option is Snake and not standard:
                standard = bm.result


if __name__ == '__main__':
    parser = build_cli_parser()
    parser.add_argument('-p', '--path', help='path to save the datafile', type=str, default=None)
    parser.add_argument('--prof', '--profile', dest='profile', action='store_const', const='True', 
        default=False, help='enable profiling')

    run(parser.parse_args())
