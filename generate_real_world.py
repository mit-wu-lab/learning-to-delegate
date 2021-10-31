from util import *

def generate_i(gen_args):
    (i, (xys, demands, capacity, ptype)), seed, args = gen_args
    assert ptype == args.ptype
    np.random.seed(seed)
    start_time = time()
    if args.n_nodes > 0:
        n_nodes_original = len(xys) - 1
        print(f'Generating problem {i} from problem of size {n_nodes_original}...')
        idxs = 1 + np.random.choice(n_nodes_original, size=args.n_nodes, replace=False)
        xys = np.concatenate((xys[:1], xys[idxs].astype(np.float32)), axis=0)
        demands = demands[idxs]
    xymin = xys.min(axis=0, keepdims=True)
    xymax = xys.max(axis=0, keepdims=True)
    xys = (xys - xymin) / (xymax - xymin)
    if not args.keep_demands:
        demands = np.random.randint(args.min_demand, args.max_demand, size=1 + args.n_nodes)
        demands[0] = 0
        capacity = args.capacity

    init_routes = solve_init(xys, demands, args, capacity=capacity, ptype=ptype)
    p = VRFullProblem(xys, demands, capacity, init_routes)

    total_time = time() - start_time
    print(f'Problem {i} took {total_time:.4f} seconds', flush=True)
    return xys, demands, capacity, p.route_dists, pack_routes(init_routes), total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('real_dir', type=Path)
    parser.add_argument('save_dir', type=Path)
    parser.add_argument('partition', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('--glob', type=str, default='*.txt')
    parser.add_argument('--n_instances_per_example', type=int, default=None)
    parser.add_argument('--keep_demands', action='store_true')
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_lkh_trials', type=int, default=100)
    parser.add_argument('--time_threshold', type=int, default=None) # HGS
    parser.add_argument('--min_demand', type=int, default=1, help='Inclusive')
    parser.add_argument('--max_demand', type=int, default=10, help='Exclusive')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--n_cpus', type=int, default=40)
    parser.add_argument('--n_process', type=int, default=None)
    parser.add_argument('--n_threads_per_process', type=int, default=None)
    parser.add_argument('--solver', type=str, choices=['LKH', 'HGS'], default='LKH')
    parser.add_argument('--full_solver_init', action='store_true')
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True)
    args.ptype = 'CVRP'

    save_path = args.save_dir / f'problems_{args.partition}.npz'
    if save_path.exists():
        print(f'Already generated {save_path}, quitting')
        exit()
    print(f'Generating {args.n_instances_per_example} instances per example, for the following examples:')
    src_paths = sorted(args.real_dir.glob(args.glob))
    src_data = [read_vrplib_input(p) for p in src_paths]

    tasks = [(i * args.n_instances_per_example + j, d) for i, d in enumerate(src_data) for j in range(args.n_instances_per_example)]
    print(f'Running {len(tasks)} total tasks')

    results = multiprocess(generate_i, list(zip(tasks,
        np.random.randint(np.iinfo(np.int32).max, size=len(tasks)),
        [args] * len(tasks)
    )), cpus=args.n_process or (args.n_cpus - 1) // args.n_clusters + 1)
    xys, demands, capacities, route_dists, init_tours, times = zip(*results)
    xys = pad_each(xys)
    demands = pad_each(demands)
    route_dists = pad_each(route_dists)
    init_tours = pad_each(init_tours)
    np.savez(save_path, nodes=xys, demands=demands, capacities=capacities, dists=route_dists, routes=init_tours, times=np.array(times))
