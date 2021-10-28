from util import *

parser = argparse.ArgumentParser()
parser.add_argument('path_or_globs', type=Path, nargs='+')
parser.add_argument('output', type=Path)

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.output.name.endswith('_subproblems.npz')
    here = Path('.')
    npzs = [np.load(path) for glob in args.path_or_globs for path in glob.parent.glob(glob.name)]

    n_nodes = [len(npz['xs']) for npz in npzs]
    offsets = np.cumsum(n_nodes) - n_nodes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: np.concatenate([(npz[key] + o) if key == 'offsets' else npz[key] for npz, o in zip(npzs, offsets)]) for key in npzs[0].files})