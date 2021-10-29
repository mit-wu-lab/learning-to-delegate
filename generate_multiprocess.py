from util import *

def generate_i(gen_args):
    i, j, (nodes, demands, capacity, routes, pkwargs), args = gen_args
    save_ij = args.partition_dir / f'{i}{f"_{j}" if j else ""}.npz'
    if save_ij.exists():
        print(f'Skipping problem {i} run {j}...', flush=True)
        return
    print(f'Generating problem {i} run {j}...', flush=True)
    start_time = time()

    class AC(ActionCallback):
        if args.sample:
            def action_order(self, p):
                return np.random.permutation(args.n_subproblems or len(p.routes))
        elif args.max_min_dist:
            def __init__(self, args):
                super().__init__(args)
                self.chosen_xys = []

            def action_order(self, p):
                if len(self.chosen_xys) == 0:
                    return np.argsort(((p.route_centroids[p.action_centroids] - p.depot) ** 2).sum(axis=1))
                return np.argsort(((np.expand_dims(p.route_centroids[p.action_centroids], axis=1) - np.expand_dims(self.chosen_xys, axis=0)) ** 2).sum(axis=2).min(axis=1))[::-1]

            def feedback_fn(self, actions, subproblems, feedbacks):
                centroids = self.last_p.route_centroids[p.action_centroids][actions]
                if self.n_repeats[-1]:
                    self.chosen_xys[-len(actions):] = centroids
                else:
                    self.chosen_xys.extend(centroids)
        elif args.min_count:
            def action_order(self, p):
                if p.node_counter.sum() == 0:
                    return np.argsort(((p.route_centroids[p.action_centroids] - p.depot) ** 2).sum(axis=1))
                return np.argsort([p.node_counter[ri].sum() for ri in p.routes[p.action_centroids]])
        else:
            action_fn = feedback_fn = kwargs_fn = None

    cb = AC(args)

    pk_default = {}
    if args.ptype == 'CVRPTW':
        pk_default.update(window_distance_scale=args.window_distance_scale)
    save_beam_search(save_ij,
        nodes, demands, capacity, routes,
        args, pkwargs=pkwargs, n_cpus=args.n_threads_per_process or 1,
        action_fn=cb.action_fn, feedback_fn=cb.feedback_fn, kwargs_fn=cb.kwargs_fn
    )
    print(f'Problem {i} run {j} took {time() - start_time:.4f} seconds', flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('partition', type=str, choices=['train', 'val', 'test'])
parser.add_argument('--partition_suffix', type=str, default='')
parser.add_argument('--save_dir', type=Path, default=Path('save'))
parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
parser.add_argument('--window_distance_scale', type=float, default=0)
parser.add_argument('--index_start', type=int, default=None)
parser.add_argument('--index_end', type=int, default=None)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--max_min_dist', action='store_true')
parser.add_argument('--min_count', action='store_true')
parser.add_argument('--beam_width', type=int, default=3)
parser.add_argument('--generate_depth', type=int, default=30)
parser.add_argument('--n_subproblems', type=int, default=None)
parser.add_argument('--subproblem_temperature', type=float, default=0.0)
parser.add_argument('--n_route_neighbors', type=int, default=15)
parser.add_argument('--n_lkh_trials', type=int, default=50)  # LKH
parser.add_argument('--time_threshold', type=int, default=30)  # HGS
parser.add_argument('--improve_threshold', type=float, default=-float('inf'))
parser.add_argument('--detect_duplicate', action='store_true')
parser.add_argument('--no_cache', action='store_true')
parser.add_argument('--n_cpus', type=int, default=16)
parser.add_argument('--n_process', type=int, default=None)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--n_threads_per_process', type=int, default=None)
parser.add_argument('--init_tour', action='store_true')
parser.add_argument('--double_lkh', action='store_true')
parser.add_argument('--solver', type=str, choices=['LKH', 'HGS'], default='LKH')

def get_partition(args):
    partition_name = diff_args(args, parser, args.partition, window_distance_scale='wscale', n_route_neighbors='routeneighbors', beam_width='beam', sample='sample', max_min_dist='maxmindist', min_count='mincount', generate_depth='depth', improve_threshold='improve', detect_duplicate='nodup', no_cache='nocache', double_lkh='double', init_tour='init', time_threshold='hgsthres', n_subproblems='nsubp', subproblem_temperature='subptemp')
    return args.dataset_dir / partition_name

if __name__ == '__main__':
    args = parser.parse_args()
    args.partition += args.partition_suffix
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    args.partition_dir = get_partition(args)
    args.partition_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created partition {args.partition_dir}', flush=True)

    if args.solver == 'HGS':
        assert args.ptype == 'CVRP' and not args.init_tour

    nodes, demands, capacities, dists, routes, pkwargs = load_problems(args.save_dir / f'problems_{args.partition}.npz')
    pkwargs = {k: pkwargs[k] for k in dict(CVRP=[], CVRPTW=['window', 'service_time'], VRPMPD=['is_pickup'])[args.ptype]}

    args.index_start = args.index_start or 0
    args.index_end = args.index_end or len(nodes)
    print(f'Running multiprocess generation from problem index {args.index_start} to {args.index_end} with {args.n_runs} runs each', flush=True)
    prob_idxs = np.repeat(np.arange(args.index_start, args.index_end), args.n_runs)
    run_idxs = np.tile(np.arange(args.n_runs), args.index_end - args.index_start)
    p_args = [(nodes[i], demands[i], capacities[i], routes[i], {k: v[i] for k, v in pkwargs.items()}) for i in prob_idxs]
    multiprocess(generate_i, list(zip(prob_idxs, run_idxs, p_args, [args] * len(prob_idxs))), cpus=args.n_process or args.n_cpus)