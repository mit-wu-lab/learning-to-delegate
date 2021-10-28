from util import *

def preprocess(prep_args):
    i, args = prep_args
    s = np.load(args.partition_dir / f'{i}.npz')
    ps = reconstruct_data(s['xys'], s['demands'], s['capacity'], unpack_routes(s['routes']), s['transitions'], s['actions'], s['lkh_dists'], s['lkh_routes'], ptype=args.ptype, pkwargs=s, n_route_neighbors=args.n_route_neighbors, generate_depth=args.supervise_depth or args.generate_depth)
    ps = [p for p in ps if hasattr(p, 'labels')]
    node_features = VRFullProblem.process_node_counts(ps[0].get_node_features(), ps[0].ptype, use_count=False).astype(np.float16)
    idx2pas = [[] for _ in range(len(s['lkh_dists']))]
    for p in ps:
        for a, idx in enumerate(p.lkh_idxs):
            idx2pas[idx].append((p, a))

    if args.supervise_depth is None:
        assert all(len(ps) for ps in idx2pas)

    all_node_idxs = []
    statistics = []
    lkh_dists = []
    prev_dists = []
    for idx, (pas, lkh_dist) in enumerate(zip(idx2pas, s['lkh_dists'])):
        if len(pas) == 0:
            continue
        subps = [p.get_subproblem(a, do_init_routes=True, n_route_neighbors=args.n_route_neighbors) for p, a in pas]
        prev_dists.append(max(subp.total_dist for subp in subps))
        lkh_dists.append(lkh_dist)
        statistics.append(subps[0].get_features())
        all_node_idxs.append(subps[0].node_idxs)
    if args.statistics:
        return np.array(statistics), np.array(lkh_dists), np.array(prev_dists)
    n_subp_nodes = [len(node_idxs) for node_idxs in all_node_idxs]
    return node_features, np.concatenate(all_node_idxs), np.array(n_subp_nodes), np.array(lkh_dists), np.array(prev_dists)

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('partitions', type=str, nargs='+')
parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
parser.add_argument('--beam_width', type=int, default=3)
parser.add_argument('--n_route_neighbors', type=int, default=15)
parser.add_argument('--generate_depth', type=int, default=30)
parser.add_argument('--time_threshold', type=int, default=30)
parser.add_argument('--supervise_depth', type=int, default=None)
parser.add_argument('--statistics', action='store_true')
parser.add_argument('--n_cpus', type=int, default=40)

if __name__ == '__main__':
    args = parser.parse_args()
    for partition in args.partitions:
        partition_name = diff_args(args, parser, partition, n_route_neighbors='routeneighbors', beam_width='beam', generate_depth='depth', time_threshold='hgsthres')
        depth_label = {'supervise_depth' if args.supervise_depth else 'generate_depth': 'depth'}
        output_name = diff_args(args, parser, partition, n_route_neighbors='routeneighbors', beam_width='beam', **depth_label, time_threshold='hgsthres') + ('_subproblem_statistics' if args.statistics else '_subproblems')
        args.partition_dir = partition_dir = args.dataset_dir / partition_name
        if not partition_dir.exists():
            print(f'Skipping {partition_dir} because it doesn\'t exist')
            exit()

        n_instances = len([x for x in partition_dir.glob('*.npz') if re.match('\d+.npz', x.name)])
        print(f'Preprocessing {n_instances} instances from partition {partition_name} into {output_name}.npz', flush=True)

        out_path = args.dataset_dir / f'{output_name}.npz'
        if out_path.exists():
            print(f'Skipping because {out_path} exists already')
            continue
        if args.statistics:
            statistics, lkh_dists, prev_dists = map(np.concatenate, zip(*multiprocess(preprocess, list(zip(range(n_instances), [args] * n_instances)), cpus=args.n_cpus)))
            np.savez(out_path, statistics=statistics, lkh_dists=lkh_dists, prev_dists=prev_dists)
        else:
            node_features, all_node_idxs, n_subp_nodes, lkh_dists, prev_dists = zip(*multiprocess(preprocess, list(zip(range(n_instances), [args] * n_instances)), cpus=args.n_cpus))

            n_nodes = np.array([nf.shape[0] for nf in node_features])
            prob_offsets = np.cumsum(n_nodes) - n_nodes

            node_offsets = [[o] * len(ns) for o, ns in zip(prob_offsets, n_subp_nodes)]

            node_features, node_offsets, all_node_idxs, n_subp_nodes, lkh_dists, prev_dists = map(np.concatenate, [node_features, node_offsets, all_node_idxs, n_subp_nodes, lkh_dists, prev_dists])

            np.savez(out_path, xs=node_features, offsets=node_offsets, subp_node_idxs=all_node_idxs, n_subp_nodes=n_subp_nodes, lkh_dists=lkh_dists, prev_dists=prev_dists)
        print(f'Saved {out_path}')
