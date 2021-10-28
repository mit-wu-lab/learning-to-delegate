import itertools
from util import *

def preprocess(prep_args):
    i, args = prep_args
    s = np.load(args.partition_dir / f'{i}.npz')
    ps = reconstruct_data(s['xys'], s['demands'], s['capacity'], unpack_routes(s['routes']), s['transitions'], s['actions'], s['lkh_dists'], s['lkh_routes'], ptype=args.ptype, pkwargs=s, n_route_neighbors=args.n_route_neighbors, generate_depth=args.supervise_depth or args.generate_depth)
    ps = [p for p in ps if hasattr(p, 'labels')]
    if args.improve_threshold is not None:
        ps = [p for p in ps if getattr(p, 'delta', -args.improve_threshold) <= -args.improve_threshold]
    return [(
        p.get_node_features().astype(np.float16),
        len(p.routes),
        p.get_route_features().astype(np.float16),
        pack_routes(p.routes).astype(np.uint16),
        p.route_neighbors[:, :args.n_route_neighbors].astype(np.uint16),
        p.labels.astype(np.float16),
        p.unique_mask.astype(np.bool)
    ) for p in ps]

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('partitions', type=str, nargs='+')
parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
parser.add_argument('--beam_width', type=int, default=3)
parser.add_argument('--n_route_neighbors', type=int, default=15)
parser.add_argument('--generate_depth', type=int, default=30)
parser.add_argument('--supervise_depth', type=int, default=None)
parser.add_argument('--improve_threshold', type=float, default=None)
parser.add_argument('--n_cpus', type=int, default=40)
parser.add_argument('--init_tour', action='store_true')
parser.add_argument('--double_lkh', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    for partition in args.partitions:
        partition_name = diff_args(args, parser, partition, n_route_neighbors='routeneighbors', beam_width='beam', generate_depth='depth', double_lkh='double', init_tour='init')
        depth_label = {'supervise_depth' if args.supervise_depth else 'generate_depth': 'depth'}
        output_name = diff_args(args, parser, partition, n_route_neighbors='routeneighbors', beam_width='beam', **depth_label, double_lkh='double', init_tour='init')
        args.partition_dir = partition_dir = args.dataset_dir / partition_name
        if not partition_dir.exists():
            print(f'Skipping {partition_dir} because it doesn\'t exist')
            exit()

        n_instances = len([x for x in partition_dir.glob('*.npz') if re.match('\d+.npz', x.name)])
        print(f'Preprocessing {n_instances} instances from partition {partition_name} into {output_name}.npz', flush=True)

        data = zip(*multiprocess(preprocess, list(zip(range(n_instances), [args] * n_instances)), cpus=args.n_cpus))
        xs, n_routes, ts, tours, route_neighbors, labels, unique_masks = zip(*itertools.chain(*data))

        xs, n_routes = map(np.array, (xs, n_routes))
        max_n_routes = n_routes.max()
        max_n_tour = max(len(t) for t in tours)
        ts, route_neighbors, labels, unique_masks, tours = map(pad_each,
            (ts, route_neighbors, labels, unique_masks, tours),
            [max_n_routes] * 4 + [max_n_tour]
        )
        np.savez(args.dataset_dir / f'{output_name}.npz', xs=xs, n_routes=n_routes, ts=ts, routes=tours, route_neighbors=route_neighbors, labels=labels, unique_masks=unique_masks)
        print(f'Saved {args.dataset_dir}/{output_name}.npz')
