from util import *

def generate_i(gen_args):
    i, j, (nodes, demands, capacity, routes, pkwargs), args = gen_args
    dir_ij = args.partition_dir / f'{i}{f"_{j}" if j else ""}'
    if dir_ij.exists():
        print(f'Skipping problem {i} run {j}...', flush=True)
        return
    dir_ij.mkdir()
    print(f'Generating problem {i} run {j}...', flush=True)
    start_time = time()

    run_lkh(*VRFullProblem(nodes, demands, capacity, routes, ptype=args.ptype, pkwargs=pkwargs).get_lkh_args(directory=dir_ij, init=args.init_tour, max_trials=args.n_lkh_trials), seed=j)
    print(f'Problem {i} run {j} took {time() - start_time:.4f} seconds', flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('partition', type=str, choices=['train', 'val', 'test'])
parser.add_argument('--partition_suffix', type=str, default='')
parser.add_argument('--save_dir', type=Path, default=Path('save'))
parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
parser.add_argument('--index_start', type=int, default=None)
parser.add_argument('--index_end', type=int, default=None)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--n_lkh_trials', type=int, default=50)
parser.add_argument('--n_cpus', type=int, default=16)
parser.add_argument('--init_tour', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    args.partition += args.partition_suffix
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    partition_name = args.partition
    args.partition_dir = args.dataset_dir / partition_name
    args.partition_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created partition {args.partition_dir}', flush=True)

    nodes, demands, capacities, dists, routes, pkwargs = load_problems(args.save_dir / f'problems_{args.partition}.npz')

    args.index_start = args.index_start or 0
    args.index_end = args.index_end or len(nodes)
    print(f'Running full LKH from problem index {args.index_start} to {args.index_end} with {args.n_runs} runs each', flush=True)
    prob_idxs = np.repeat(np.arange(args.index_start, args.index_end), args.n_runs)
    run_idxs = np.tile(np.arange(args.n_runs), args.index_end - args.index_start)
    p_args = [(nodes[i], demands[i], capacities[i], routes[i], {k: v[i] for k, v in pkwargs.items()}) for i in prob_idxs]
    multiprocess(generate_i, list(zip(prob_idxs, run_idxs, p_args, [args] * len(run_idxs))), cpus=args.n_cpus)