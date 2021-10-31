from util import *

parser = argparse.ArgumentParser()
parser.add_argument('path_or_globs', type=Path, nargs='+')
parser.add_argument('--statistics', action='store_true')
parser.add_argument('output', type=Path)

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.output.name.endswith('_subproblems_statistics.npz' if args.statistics else '_subproblems.npz')
    here = Path('.')
    npzs = [np.load(path) for glob in args.path_or_globs for path in glob.parent.glob(glob.name)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.statistics:
        np.savez(args.output, **{key: np.concatenate([npz[key] for npz in npzs]) for key in npzs[0].files})
    else:
        n_nodes = [len(npz['xs']) for npz in npzs]
        offsets = np.cumsum(n_nodes) - n_nodes
        np.savez(args.output, **{key: np.concatenate([(npz[key] + o) if key == 'offsets' else npz[key] for npz, o in zip(npzs, offsets)]) for key in npzs[0].files})