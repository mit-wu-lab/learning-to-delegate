from util import *

def clustered_xys(args, center_depot=False, max_xy=1):
    uniform_frac = 0.5 if args.mixed else 0.0
    n_uniform = int((args.n_nodes - args.n_c) * uniform_frac)
    n_clustered = args.n_nodes - args.n_c - n_uniform
    uniform_locs = np.random.uniform(0, max_xy, size=(n_uniform, 2))

    assert args.n_c < args.n_nodes
    centers = np.random.uniform(0.2, max_xy - 0.2, size=(args.n_c, 2))

    n_clustered_samples = 0
    all_clustered_locs = []
    while n_clustered_samples < n_clustered:
        center_locs = centers[np.random.randint(len(centers), size=2 * (n_clustered - n_clustered_samples))]
        cluster_locs = np.random.normal(center_locs, args.std_cluster)
        cluster_locs = cluster_locs[(cluster_locs >= 0).all(axis=1) & (cluster_locs < max_xy).all(axis=1)]
        all_clustered_locs.append(cluster_locs)
        n_clustered_samples += len(cluster_locs)
    cluster_locs = np.concatenate(all_clustered_locs)[:n_clustered]
    xys = np.vstack((centers, uniform_locs, cluster_locs))
    if center_depot:
        depot = np.mean(xys, axis=0, keepdims=True)
    else:
        min_x, min_y = np.clip(xys.min(axis=0) - 0.1, 0, max_xy)
        max_x, max_y = np.clip(xys.max(axis=0) + 0.1, 0, max_xy)
        depot = np.array([[np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]])
    return np.vstack((depot, xys))

def generate_problem(args, init=None):
    if init:
        xys, demands, capacity, pkwargs = init
    else:
        xys = clustered_xys(args) if args.n_c else np.random.uniform(0, 1, size=(1 + args.n_nodes, 2))
        demands = np.random.randint(args.min_demand, args.max_demand, size=1 + args.n_nodes)
        demands[0] = 0
        pkwargs = {}
        if args.ptype == 'CVRPTW':
            windows = np.random.uniform(0, 1, size=(1 + args.n_nodes, 2))
            dist_depot = cdist(xys[1:], [xys[0]]).flatten()
            windows[0, 0], windows[0, 1] = depot_start, depot_end = 0, 3  # np.max(dist_depot) * 2
            time_centers = np.random.uniform(depot_start + dist_depot, depot_end - dist_depot - args.service_time)
            time_half_width = np.random.uniform(args.service_time / 2, args.max_window_width, size=(args.n_nodes))

            # start time: 0 - 2; end time: 1 - 3
            windows[1:, 0] = np.clip(time_centers - time_half_width, depot_start, depot_end)
            windows[1:, 1] = np.clip(time_centers + time_half_width, depot_start, depot_end)
            pkwargs['window'] = windows
            pkwargs['service_time'] = args.service_time
        elif args.ptype == 'VRPMPD':
            pkwargs['is_pickup'] = is_pickup = np.zeros_like(demands, dtype=np.bool)
            is_pickup[1::args.pickup_every] = True

    init_routes = solve_init(xys, demands, args, pkwargs=pkwargs)
    return VRFullProblem(xys, demands, args.capacity, init_routes, ptype=args.ptype, pkwargs=pkwargs)

def generate_i(gen_args):
    i, seed, args, init = gen_args
    np.random.seed(seed)
    start_time = time()
    print(f'Generating problem {i}...')

    p = generate_problem(args, init)

    total_time = time() - start_time
    print(f'Problem {i} took {total_time:.4f} seconds')
    return p.xys, p._demands, p.capacity, p.route_dists, pack_routes(p.routes), p.pkwargs, total_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=Path)
    parser.add_argument('partition', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('--n_c', type=int, default=0, help='Number of city clusters in the problem instance')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--std_cluster', type=float, default=0.07, help='Standard deviation for normal distribution of city clusters')
    parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
    parser.add_argument('--n_instances', type=int, default=None)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_lkh_trials', type=int, default=100)
    parser.add_argument('--min_demand', type=int, default=1, help='Inclusive')
    parser.add_argument('--max_demand', type=int, default=10, help='Exclusive')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--service_time', type=float, default=0.02)
    parser.add_argument('--max_window_width', type=float, default=1.5)
    parser.add_argument('--pickup_every', type=int, default=2)
    parser.add_argument('--n_cpus', type=int, default=40)
    parser.add_argument('--n_process', type=int, default=None)
    parser.add_argument('--n_threads_per_process', type=int, default=None)
    parser.add_argument('--solver', type=str, choices=['LKH', 'HGS'], default='LKH')
    parser.add_argument('--naive_init', action='store_true')
    parser.add_argument('--full_solver_init', action='store_true')
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.n_instances = args.n_instances or (2000 if args.partition == 'train' else 40)

    partition = args.partition
    ref_path = ref_problems = None
    save_path = args.save_dir / f'problems_{partition}.npz'
    if args.naive_init or args.full_solver_init or args.n_lkh_trials != parser.get_default('n_lkh_trials'):
        ref_path = save_path
        assert ref_path.exists(), f'If using --naive_init or --full_solver_init or --n_lkh_trials, must have already used the default init to generate {ref_path}'
        partition_name = diff_args(args, parser, partition, naive_init='initnaive', full_solver_init='initfull', n_lkh_trials='lkh')
        save_path = args.save_dir / f'problems_{partition_name}.npz'
    if save_path.exists():
        print(f'Already generated {save_path}, quitting', flush=True)
        exit()
    print(f'Generating to {save_path}', flush=True)
    if ref_path:
        print(f'Generating {args.n_instances} {args.ptype} initial solutions for previous problems from {ref_path}', flush=True)
        ref_problems = np.load(ref_path)
        nodes, demands, capacities = ref_problems['nodes'], ref_problems['demands'], ref_problems['capacities']
        assert demands.shape == (args.n_instances, args.n_nodes + 1)
        pkwarg_keys = dict(CVRP=[], CVRPTW=['window', 'service_time'], VRPMPD=['is_pickup'])[args.ptype]
        pkwargs = [{k: ref_problems[k][i] for k in pkwarg_keys} for i in range(args.n_instances)]
    else:
        print(f'Generating {args.n_instances} {args.ptype} instances from {"uniform distribution" if args.n_c == 0 else f"mixed distribution with {args.n_c} city clusters" if args.mixed else f"clustered distribution with {args.n_c} city_clusters"}, each with {args.n_clusters} radial sections to run LKH subsolver on', flush=True)

    results = multiprocess(generate_i, list(zip(
        range(0, args.n_instances),
        np.random.randint(np.iinfo(np.int32).max, size=args.n_instances),
        [args] * args.n_instances,
        zip(nodes, demands, capacities, pkwargs) if ref_problems else [None] * args.n_instances,
    )), cpus=args.n_process or (args.n_cpus - 1) // args.n_clusters + 1)

    xys, demands, capacities, route_dists, init_tours, pkwargs, times = zip(*results)
    route_dists = pad_each(route_dists)
    init_tours = pad_each(init_tours)
    pkwargs = {k: np.array([pk[k] for pk in pkwargs]) for k in pkwargs[0]}
    np.savez(save_path, nodes=xys, demands=demands, capacities=capacities, dists=route_dists, routes=init_tours, times=np.array(times), **pkwargs)
