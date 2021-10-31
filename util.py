import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
import random
import re
import sys
import tempfile
from time import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import yaml
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from subprocess import check_call, check_output
from urllib.parse import urlparse

TMPDIR = Path('/dev/shm' if os.path.exists('/dev/shm') else os.environ['TMPDIR'])

def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.4.tgz"):
    cwd = os.path.abspath('lkh3')
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable), f'Cannot find LKH-3 executable file at {executable}'
    return os.path.abspath(executable)

def get_hgs_executable(url="https://github.com/vidalt/HGS-CVRP.git"):
    cwd = os.path.abspath('hgs')
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(f"git clone {url}", cwd=cwd, shell=True)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make test", cwd=os.path.join(filedir, "Program"), shell=True)

    executable = os.path.join(os.path.join(filedir, "Program"), "genvrp")
    assert os.path.isfile(executable), f'Cannot find HGS executable file at {executable}'
    return os.path.abspath(executable)

executable = get_lkh_executable()
hgs_executable = get_hgs_executable()

def write_lkh_params(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 500,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0,
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))

def write_lkh_tours(tours, n_jobs, name, tour_filename, cost=None):
    n_veh = len(tours)
    with open(tour_filename, 'w') as f:
        f.write('TYPE : TOUR\n')
        f.write('DIMENSION : {}\n'.format(n_jobs + n_veh))
        f.write('TOUR_SECTION\n')
        for tour_idx, tour in enumerate(tours):  # 1: depot, 2 to n_jobs + 1: jobs, n_jobs + 1+1 - : depot
            f.write('{}\n'.format(1 if tour_idx == 0 else n_jobs + n_veh + 1 - tour_idx))
            for a in tour:  # depot starts at 1 (originally 0), jobs start at 2 (originally 1)
                f.write('{}\n'.format(a+1))
        f.write('-1\n')
        f.write('EOF')

def write_hgs_tours(tours, tour_filename, cost):  # cost can't be None
    with open(tour_filename, 'w') as f:
        for tour_idx, tour in enumerate(tours):
            f.write('Route #{}: {}\n'.format(tour_idx+1, " ".join([str(a+1) for a in tour])))
        f.write('Cost {}\n'.format(cost))

def read_vrplib_input(filename):
    with open(filename, 'r') as f:
        coordinates = []
        demands = []
        depots = []
        append = None
        for line in f:
            if line.startswith('NAME'):
                continue
            elif line.startswith('COMMENT'):
                continue
            elif line.startswith('TYPE'):
                ptype = line.split(':')[1].lstrip().rstrip()
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(':')[1].lstrip().rstrip())
            elif line.startswith('NODE_COORD_SECTION'):
                append = coordinates
            elif line.startswith('DEMAND_SECTION'):
                append = demands
            elif line.startswith('DEPOT_SECTION'):
                append = depots
            elif line.startswith('EOF') or line.startswith('END'):
                break
            elif append is not None:
                append.append(line.rstrip().split())
        coordinates = np.array(coordinates).astype(int)[:, 1:]
        demands = np.array(demands).astype(int)[:, 1]
        if len(depots):
            depots = np.array(depots).astype(int)
            assert depots[0] == 1 and depots[1] == -1
        assert demands[0] == 0
    return coordinates, demands, capacity, ptype

def read_vrplib_solution(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    return tour

def write_vrplib(filename, depot, loc, demand, capacity, ptype, pkwargs, grid_size, name="problem", scale=100000):
    to_int = lambda arr: (np.array(arr) / grid_size * scale + 0.5).astype(int)
    with open(filename, 'w') as f:
        f.write("\n".join(
            f"{k} : {v}" for k, v in (
                ("NAME", name),
                ("COMMENT", "None"), # HGS assumes a comment line follows NAME
                ("TYPE", ptype),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ))
        if ptype == "CVRPTW":
            f.write("\n")
            f.write(f"SERVICE_TIME : {to_int(pkwargs['service_time'])}")
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join(
            f"{i + 1}\t{x}\t{y}" # VRPlib does not take floats
            for i, (x, y) in enumerate(itertools.chain([to_int(depot)], to_int(loc)))
        ))
        if ptype == "VRPMPD":
            f.write("\n")
            f.write("PICKUP_AND_DELIVERY_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t0\t0\t10000000\t0\t{d * p}\t{d * (not p)}"
                for i, (d, p) in enumerate(zip(itertools.chain([0], demand), pkwargs['is_pickup']))
            ))
        else:
            f.write("\n")
            f.write("DEMAND_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t{d}" for i, d in enumerate(itertools.chain([0], demand))
            ))
        if ptype == "CVRPTW":
            f.write("\n")
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join(
                f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate(to_int(pkwargs['window']))
            ))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")

def read_hgs(filename, n):
    tour = []
    num_routes = 0
    with open(filename, 'r') as f:
        ended = False
        for line in f:
            if line.startswith("Cost"):
                ended = True
                cost = int(float(line.strip().split()[1]))
            if line.startswith("Time"):
                time = float(line.strip().split()[1])

            if not ended:
                l = line.strip().split(":")
                tour.append(0)
                tour.extend(map(int, l[1].split()))
                num_routes += 1

    tour = np.array(tour).astype(int) # depot is 0 and other nodes start with 1 with HGS format
    assert len(tour) - num_routes == np.max(tour) == n

    return tour

yaml_types = (str, int, float, bool, type(None))

def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_problems(path):
    problems = np.load(path)
    keys = ['nodes', 'demands', 'capacities', 'dists', 'routes']
    rets = [problems[k] for k in keys]
    rets[-1] = [unpack_routes(rs) for rs in rets[-1]]
    pkwargs = {k: problems[k] for k in set(problems.files) - set(keys)}
    return *rets, pkwargs

def pad_to(array, *lengths, **kwargs):
    lengths = lengths + array.shape[len(lengths):]
    return np.pad(array, [(0, n_pad - n) for n_pad, n in zip(lengths, array.shape)], **kwargs)

def pad_each(array, length=None):
    length = length or max(len(row) for row in array)
    return np.array([pad_to(row, length) for row in array])

def pack_routes(routes, max_len_tour=None, left_pad=0, right_pad=None):
    routes = np.concatenate([r_ for r in routes for r_ in (r, [0])])
    max_len_tour = max_len_tour or len(routes)
    right_pad = (max_len_tour - len(routes) - left_pad) if right_pad is None else right_pad
    return np.pad(routes, (left_pad, right_pad)).astype(np.int16)

def unpack_routes(routes): # Split zero-padded concatenated routes into separate arrays
    routes = [r[:-1] if r[-1] == 0 else r for r in np.split(routes, np.where(routes == 0)[0] + 1) if len(r)]
    return [r for r in routes if len(r)]

def get_route_distance(route, distance_matrix):
    last_node = 0
    distance = 0.0
    for c in route:
        distance += distance_matrix[last_node][c]
        last_node = c
    distance += distance_matrix[last_node][0]
    return distance

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.imap(func, tasks))

def multithread(func, tasks, cpus=None, show_bar=True):
    bar = lambda x: tqdm(x, total=len(tasks)) if show_bar else x
    if cpus == 1 or len(tasks) == 1:
        return [func(t) for t in bar(tasks)]
    with ThreadPool(cpus or os.cpu_count()) as pool:
        return list(bar(pool.imap(func, tasks)))

class Namespace(dict):
    def __init__(self, *args, **kwargs):
        kvs = dict()
        for a in args:
            if type(a) is str:
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        return self

    def new(self, *args, **kwargs):
        return Namespace({**self, **Namespace(*args, **kwargs)})

def diff_args(args, defaults, base=None, **labels):
    defaults = {x.dest: x.default for x in defaults._actions} if isinstance(defaults, argparse.ArgumentParser) else defaults
    segments = [base] if base else []
    for arg_name, arg_label in labels.items():
        arg_value = getattr(args, arg_name)
        if arg_value != defaults[arg_name]:
            segments.append(f'{arg_label}{"" if isinstance(arg_value, bool) else arg_value}'.replace('_', ''))
    return '_'.join(segments)

def diff_args_cmd(args, parser):
    args_strs = []
    for x in parser._actions:
        if x.dest == 'help' or getattr(args, x.dest) == parser.get_default(x.dest): continue
        if len(x.option_strings):
            s = x.option_strings[0]
            if not isinstance(args[x.dest], bool):
                s = f'{s} {args[x.dest]}'
        else:
            s = str(args[x.dest])
        args_strs.append(s)
    args_str = ' '.join(args_strs)
    return args_str

def compute_angle(xys):
    depot, locs = xys[0], xys[1:]
    rand_xy = locs[np.random.choice(len(locs))]
    ang_locs = np.arctan2(*(locs - depot).T)
    ang_rand = np.arctan2(*(rand_xy - depot).T)
    return (ang_locs - ang_rand) % (2 * np.pi)

def solve_init(xys, demands, args, capacity=None, ptype=None, pkwargs={}, seed=0):
    capacity = capacity or args.capacity
    ptype = ptype or args.ptype
    if getattr(args, 'naive_init', False):
        assert ptype == 'CVRP'
        sum_demand = 0
        routes = [[]]
        for i, demand in enumerate(demands):
            sum_demand += demand
            if sum_demand > capacity:
                sum_demand = demand
                routes.append([])
            routes[-1].append(i)
    elif getattr(args, 'full_solver_init', False):
        try:
            p = VRProblem(xys, demands, capacity, ptype=ptype, pkwargs=pkwargs)
            if args.solver == 'LKH':
                tour, _ = run_lkh(*p.get_lkh_args(max_trials=args.n_lkh_trials))
            elif args.solver == 'HGS':
                tour, _ = run_hgs(*p.get_hgs_args(time_threshold=args.time_threshold), seed=seed)
            routes = unpack_routes(tour)
        except:
            print('HGS failed, falling back to naive', flush=True)
            sum_demand = 0
            routes = [[]]
            for i, demand in enumerate(demands):
                sum_demand += demand
                if sum_demand > capacity:
                    sum_demand = demand
                    routes.append([])
                routes[-1].append(i)
    else:
        angles = compute_angle(xys)
        idxs = np.array_split(1 + np.argsort(angles), args.n_clusters)
        _idxs = [np.pad(idxs_i, (1, 0)) for idxs_i in idxs]
        _N = len(xys)
        subps = [VRProblem(xys[_idxs_i], demands[_idxs_i], capacity, ptype=ptype, pkwargs={k: v if np.isscalar(v) or v.ndim == 0 else v[_idxs_i] for k, v in pkwargs.items()}) for _idxs_i in _idxs]
        if args.solver == 'LKH':
            lkh_args = [p.get_lkh_args(max_trials=args.n_lkh_trials) for p in subps]
            results = multithread(lambda x: run_lkh(*x), lkh_args, cpus=args.n_threads_per_process or min(args.n_clusters, args.n_cpus), show_bar=False)
        elif args.solver == 'HGS':
            hgs_args = [p.get_hgs_args(time_threshold=args.time_threshold) for p in subps]
            results = multithread(lambda x: run_hgs(*x, seed=seed), hgs_args,
                                cpus=args.n_threads_per_process or min(args.n_clusters, args.n_cpus), show_bar=False)
        routes = [_idxs_i[subtour] for _idxs_i, (subtour, _) in zip(_idxs, results)]
    return routes

class VRProblem:
    def __init__(self, xys, demands, capacity, node_dist_map=None, ptype='CVRP', pkwargs={}):
        self.xys = xys
        self.depot, self.locs = xys[0], xys[1:]
        if len(demands) == len(xys):
            assert demands[0] == 0
            self.demands, self._demands = demands[1:], demands
        else:
            self.demands, self._demands = demands, np.pad(demands, (1, 0))
        self.capacity = capacity
        self.ptype = ptype
        self.pkwargs = pkwargs
        self.node_dist_map = cdist(xys, xys) if node_dist_map is None else node_dist_map

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)

    def get_lkh_args(self, directory=None, init=False, max_trials=500):
        init_routes, init_dist = (self.routes, self.total_dist) if init else (None, None)
        return directory, self.depot, self.locs, self.demands, self.capacity, init_routes, init_dist, max_trials, self.ptype, self.pkwargs

    def get_hgs_args(self, directory=None, time_threshold=30):
        return directory, self.depot, self.locs, self.demands, self.capacity, time_threshold

class VRFullProblem(VRProblem):
    def __init__(self, xys, demands, capacity, routes, node_dist_map=None, route_dists=None, ptype='CVRP', pkwargs={}, node_counter=None):
        super().__init__(xys, demands, capacity, node_dist_map=node_dist_map, ptype=ptype, pkwargs=pkwargs)

        self.routes = np.array(routes, dtype=object)
        self.route_dists = np.array([get_route_distance(r, self.node_dist_map) for r in routes]) if route_dists is None else route_dists
        self.total_dist = self.route_dists.sum()

        self.route_num_nodes = np.array([len(ri) for ri in routes])
        self.route_centroids = np.array([xys[ri].mean(axis=0) for ri in routes])
        self.route_dist_map = route_dist_map = cdist(self.route_centroids, self.route_centroids)
        if ptype == 'CVRPTW' and pkwargs.get('window_distance_scale', 0) > 0:
            window = pkwargs['window']
            window_centroids = np.array([window[ri].mean(axis=0) for ri in routes])
            route_dist_map += pkwargs['window_distance_scale'] * cdist(window_centroids, window_centroids, metric='cityblock')
        self.route_neighbors = route_dist_map.argsort(axis=1)

        self.node_counter = np.zeros((len(xys),)) if node_counter is None else node_counter

        self.prev_id = None
        self.prev_action = None
        self.id = None
        self._hash = None

    def set_prev(self, prev_id, prev_action):
        self.prev_id, self.prev_action = prev_id, prev_action
        return self

    def set_id(self, id):
        self.id = id
        return self

    def set_depth(self, depth):
        self.depth = depth
        return self

    def get_cluster(self, action, n_subproblems=None, max_num_nodes=200, n_route_neighbors=15, route_only=False, temperature=None):
        n_routes = len(self.routes)
        if (action_centroids := getattr(self, 'action_centroids', None)) is None:
            if n_subproblems or temperature:
                n_subproblems = n_subproblems or n_routes
                # Deterministically generate randomized center indices with the sorted self.routes as seed
                self.action_centroids = action_centroids = np.random.RandomState(hash(self) % np.iinfo(np.uint32).max).choice(n_routes, n_subproblems, replace=bool(temperature))
            else:
                self.action_centroids = action_centroids = range(n_routes)
        i_centroid = action_centroids[action]
        if temperature:
            probs = scipy.special.softmax(-self.route_dist_map[i_centroid] / temperature) # Smaller dist = larger probability
            # Deterministically generate randomized nearest neighbors with action as the seed
            # to ensure that multiple subproblems with the same i_centroid will have different nearest neighbors
            route_idxs = np.random.RandomState(action).choice(len(probs), size=n_route_neighbors, replace=False, p=probs)
        else:
            route_idxs = self.route_neighbors[i_centroid, :n_route_neighbors]
        num_nodes = np.cumsum(self.route_num_nodes[route_idxs])
        last_route = n_route_neighbors - 1
        while num_nodes[last_route] > max_num_nodes:
            last_route -= 1
        route_idxs = np.sort(route_idxs[:last_route + 1])
        if route_only:
            return route_idxs
        node_idxs = np.sort(np.concatenate(self.routes[route_idxs]))
        return route_idxs, node_idxs

    def get_subproblem(self, action, do_init_routes=False, **kwargs):
        route_idxs, node_idxs = self.get_cluster(action, **kwargs)
        return VRSubProblem(self, action, node_idxs, route_idxs, do_init_routes)

    def get_subproblems(self, n_subproblems=None, **kwargs):
        return [self.get_subproblem(a, n_subproblems=n_subproblems, **kwargs) for a in range(n_subproblems or len(self.routes))]

    def get_subproblem_dist(self, subproblem):
        return self.route_dists[subproblem.route_idxs].sum()

    def get_subproblem_dists(self, subproblems):
        return [self.get_subproblem_dist(sp) for sp in subproblems]

    def apply_subproblem(self, subproblem):
        mask = np.ones(len(self.routes), dtype=np.bool)
        mask[subproblem.route_idxs] = False
        node_idxs = subproblem._node_idxs
        subp_routes = [node_idxs[r] for r in subproblem.routes]
        routes = np.array(list(itertools.chain(self.routes[mask], subp_routes)), dtype=object)
        route_dists = np.concatenate([self.route_dists[mask], subproblem.route_dists])
        new_node_counter = self.node_counter.copy()
        new_node_counter[subproblem.node_idxs] += 1
        return VRFullProblem(self.xys, self.demands, self.capacity, routes, node_dist_map=self.node_dist_map, ptype=self.ptype, pkwargs=self.pkwargs, node_counter=new_node_counter)

    def __hash__(self):
        if self._hash is None:
            ordered_routes = sorted([r if r[0] <= r[-1] else r[::-1] for r in self.routes], key=lambda r: r[0])
            packed_routes = pack_routes(ordered_routes)
            packed_routes.flags.writeable = False
            self._hash = int(hashlib.md5(packed_routes.data.tobytes()).hexdigest(), 16)
        return self._hash

    def get_node_features(self):
        xys = self.xys[:, :2] - self.depot
        demands = self._demands / self.capacity
        features = [xys, demands.reshape(-1, 1)]
        if self.ptype == 'CVRPTW':
            features.append(self.pkwargs['window'])
        elif self.ptype == 'VRPMPD':
            features.append(self.pkwargs['is_pickup'].reshape(-1, 1))
        features.append(self.node_counter.reshape(-1, 1))
        return np.hstack(features)

    @classmethod
    def process_node_counts(cls, x, ptype, use_count=True):
        return x if use_count else x[..., :dict(CVRP=3, CVRPTW=5, VRPMPD=4)[ptype]]

    def get_route_features(self):
        maxs = np.array([self.xys[ri].max(axis=0) for ri in self.routes]) # Do not include node 0 here
        mins = np.array([self.xys[ri].min(axis=0) for ri in self.routes]) # Do not include node 0 here
        demands = np.array([self._demands[ri].sum() for ri in self.routes]) / self.capacity
        return np.hstack([
            self.route_centroids - self.depot, maxs - self.depot, mins - self.depot,
            np.array([self.route_num_nodes / self.capacity, demands, self.route_dists]).T
        ])

    @classmethod
    def transform_features(cls, x, t, rotate=False, flip=False, perturb_node=False, perturb_route=False):
        if not (rotate or flip or perturb_node or perturb_route): return x, t
        import torch

        n_batch, n_node, _ = x.shape

        thetas = 2 * np.pi * (torch.rand if rotate else torch.zeros)(n_batch, device=x.device, dtype=x.dtype)
        s, c = thetas.sin(), thetas.cos()

        rot = torch.stack([c, -s, s, c]).view(2, 2, n_batch)
        if flip: rot[1, :, torch.randint(0, 2, (n_batch,), device=x.device, dtype=torch.bool)] *= -1

        noise_node = torch.normal(0, perturb_node, (n_batch, n_node, 2), device=x.device, dtype=x.dtype) if perturb_node else torch.zeros((n_batch, n_node, 2), device=x.device, dtype=x.dtype)
        noise_node[:, 0] = 0

        x = torch.cat([torch.einsum('ijb,bfj->bfi', rot, x[:, :, :2] + noise_node), x[:, :, 2:]], dim=2)
        if t is not None:
            max_n_routes = t.size(1)
            noise_route = torch.normal(0, perturb_route, (n_batch, max_n_routes, 2), device=x.device, dtype=x.dtype) if perturb_route else torch.zeros((n_batch, max_n_routes, 2), device=x.device, dtype=x.dtype)
            t = torch.cat([
                *(torch.einsum('ijb,bfj->bfi', rot, t[:, :, i: i + 2] + noise_route) for i in [0, 2, 4]),
                t[:, :, 6:]
            ], dim=2)
        return x, t

class VRSubProblem(VRProblem):
    def __init__(self, problem, action, node_idxs, route_idxs, do_init_routes=False):
        self._node_idxs = np.pad(node_idxs, (1, 0))
        _N = len(self._node_idxs)
        pkwargs = {k: v if np.isscalar(v) or v.ndim == 0 else v[self._node_idxs] for k, v in problem.pkwargs.items()}
        super().__init__(problem.xys[self._node_idxs], problem._demands[self._node_idxs], problem.capacity, ptype=problem.ptype, pkwargs=pkwargs)
        self.action = action
        self.problem = problem
        self.node_idxs = node_idxs
        self.route_idxs = route_idxs
        self.route_dists = problem.route_dists[route_idxs]
        self.total_dist = self.route_dists.sum()
        self._hash = None

        if do_init_routes:
            node_map = np.zeros(len(problem.xys), dtype=np.int)
            node_map[self._node_idxs] = np.arange(len(self._node_idxs))
            self.routes = np.array([node_map[r] for r in problem.routes[route_idxs]], dtype=object)

    def __hash__(self):
        if self._hash is None:
            self.node_idxs.flags.writeable = False
            self._hash = hash(self.node_idxs.data.tobytes())
        return self._hash

    def set_routes(self, routes):
        self.routes = routes
        self.route_dists = np.array([get_route_distance(r, self.node_dist_map) for r in routes])
        n_nodes = sum(len(r) for r in routes)
        assert n_nodes == len(self.node_idxs)
        self.total_dist = self.route_dists.sum()
        self.change_dist = self.total_dist - self.problem.get_subproblem_dist(self)
        return self.change_dist

    def get_features(self):
        # Return a 1D array of features
        problem = self.problem
        nodes = problem.xys[self.node_idxs]
        demands = problem._demands[self.node_idxs] / self.capacity

        depot_dists = np.linalg.norm(nodes - problem.depot, axis=1)
        centroid = nodes.mean(axis=0)
        centroid_dists = np.linalg.norm(nodes - centroid, axis=1)

        return np.array([
            len(nodes),
            *(centroid - problem.depot),
            *nodes.std(axis=0),
            *(nodes.max(axis=0) - problem.depot),
            *(nodes.min(axis=0) - problem.depot),
            depot_dists.mean(),
            depot_dists.std(),
            *(np.percentile(depot_dists, x) for x in np.linspace(0, 100, 11)),
            *(np.percentile(centroid_dists, x) for x in np.linspace(0, 100, 11)),
            demands.mean(),
            demands.std(),
        ])

def run_lkh(directory, depot, locs, demand, capacity, init_routes, init_cost, max_trials, ptype='CVRP', pkwargs={}, seed=0):
    def lkh_helper(temp):
        problem_path = os.path.join(directory, 'problem.vrp')
        init_routes_path = os.path.join(directory, 'input.tour')
        output_path = os.path.join(directory, 'output.tour')
        param_path = os.path.join(directory, 'params.par')

        params = dict(
            PROBLEM_FILE=problem_path, OUTPUT_TOUR_FILE=output_path,
            RUNS=1, MAX_TRIALS=max_trials, SEED=seed, TRACE_LEVEL=0 if temp else 1
        )
        # it seems like the CVRPTW and VRPMPD variants will set VEHICLES = 1 if not specify the number
        # (unlike CVRP that set VEHICLES > 1)
        if ptype != 'CVRP':
            params.update(VEHICLES=len(locs))
        if init_routes is not None:
            write_lkh_tours(init_routes, len(demand), 'route_init', init_routes_path, cost=init_cost)
            params.update(INITIAL_TOUR_FILE=init_routes_path, VEHICLES=len(init_routes))
        write_vrplib(problem_path, depot, locs, demand, capacity, ptype, pkwargs, 1)
        write_lkh_params(param_path, params)

        start = time()
        log_path = os.path.join(directory, 'output.log')
        if temp:
            output = check_output([executable, param_path])
        else:
            with open(log_path, 'w') as f:
                check_call([executable, param_path], stdout=f, stderr=f)
        duration = time() - start

        try:
            tour = read_vrplib_solution(output_path, n=len(demand)) # zero-separated routes
        except FileNotFoundError as e:
            print('LKH Error:')
            with open(log_path, 'r') as f:
                print(f.readlines())
            raise e
        if ptype != 'CVRP':
            # for CVRPTW and VRPMPD where we initialize with more vehicles than needed, the output tour will have consecutive 0's
            # Pack and unpack removes extra zeros before, between, and after routes
            tour = pack_routes(unpack_routes(tour))[:-1]
        return np.array(tour, dtype=np.uint16), duration

    if directory is None:
        with tempfile.TemporaryDirectory(dir=TMPDIR) as directory:
            return lkh_helper(temp=True)
    return lkh_helper(temp=False)

def run_hgs(directory, depot, locs, demand, capacity, time_threshold, seed=0):
    def hgs_helper(temp):
        problem_path = os.path.join(directory, 'problem.vrp')
        output_path = os.path.join(directory, 'output.tour')

        write_vrplib(problem_path, depot, locs, demand, capacity, 'CVRP', {}, 1, scale=10000)

        start = time()
        command = [hgs_executable, problem_path, output_path, '-seed', f'{seed}', '-t', f'{time_threshold}']
        log_path = os.path.join(directory, 'output.log')
        if temp:
            output = check_output(command)
        else:
            with open(log_path, 'w') as f:
                check_call(command, stdout=f, stderr=f)
        duration = time() - start

        try:
            tour = read_hgs(output_path, n=len(demand)) # zero-separated routes
        except:
            print('HGS Error:')
            with open(log_path, 'r') as f:
                print(f.readlines())
            raise e
        return np.array(tour, dtype=np.uint16), duration

    if directory is None:
        with tempfile.TemporaryDirectory(dir=TMPDIR) as directory:
            return hgs_helper(temp=True)
    return hgs_helper(temp=False)

def beam_search(nodes, demands, capacity, init_routes, args, pkwargs={}, n_cpus=1, action_fn=None, feedback_fn=None, show_bar=False):
    """
    beam_width=1 is the same as just taking the argmax at each step
    """
    subproblem_cache = {} # Maps subproblem to (lkh solution index, routes)

    transitions = [] # Tuples of (problem_id, action, next_problem_id)
    actions = [] # Tuples of (problem_id, action, LKH solution index)
    lkh_solutions = [] # Tuples of (routes, dist)

    problem_id = 0
    fullp = VRFullProblem(nodes, demands, capacity, init_routes, ptype=args.ptype, pkwargs=pkwargs).set_id(problem_id)
    fullps = [fullp]
    times = []
    start_time = time()

    terminate = False
    for iteration in range(args.generate_depth):
        best_problems = set()
        for fullp in fullps:
            subps = fullp.get_subproblems(n_subproblems=args.n_subproblems, do_init_routes=args.init_tour, n_route_neighbors=args.n_route_neighbors, temperature=args.subproblem_temperature)
            subps_unique = set(subps)
            if action_fn is not None:
                fullp.unique_mask = np.zeros(len(subps), dtype=np.bool)
                fullp.unique_mask[[subp.action for subp in subps_unique]] = True

                action_candidates = action_fn(fullp)
                if len(action_candidates) == 0:
                    terminate = True
                    break
                subps = [subps[a] for a in action_candidates]
                subps_unique = set(subps)
            subps_unique = list(subps_unique)
            subps_todo = [subp for subp in subps_unique if args.no_cache or subp not in subproblem_cache]

            if args.solver == 'LKH':
                tasks = [sp.get_lkh_args(max_trials=args.n_lkh_trials, init=args.init_tour) for sp in subps_todo]
                results = multithread(lambda args: run_lkh(*args), tasks, cpus=n_cpus, show_bar=show_bar)
            elif args.solver == 'HGS':
                tasks = [sp.get_hgs_args(time_threshold=args.time_threshold) for sp in subps_todo]
                results = multithread(lambda args: run_hgs(*args, seed=random.randrange(sys.maxsize) if not hasattr(args, 'seed') else args.seed), tasks, cpus=n_cpus, show_bar=show_bar)

            for i, (subp, (new_tour, duration)) in enumerate(zip(subps_todo, results)):
                subproblem_cache[subp] = (len(lkh_solutions) + i, new_tour)

            change_dists = np.array([subp.set_routes(unpack_routes(subproblem_cache[subp][1])) for subp in subps])
            actions.extend((fullp.id, subp.action, subproblem_cache[subp][0]) for subp in subps)
            lkh_solutions.extend((subp.total_dist, new_tour) for subp, (new_tour, _) in zip(subps_todo, results))
            if feedback_fn is not None:
                feedback_fn(action_candidates, subps, change_dists)

            change_dists_unique = [subp.change_dist for subp in subps_unique]
            best_idxs = np.argpartition(change_dists_unique, args.beam_width)[:args.beam_width] if action_fn is None else range(len(subps_unique))
            for idx in best_idxs:
                subp = subps_unique[idx]
                best_problems.add((fullp.total_dist + subp.change_dist, fullp, subp))
        if terminate: break

        if getattr(args, 'double_lkh', False):
            _, ps, subps = zip(*best_problems)
            if args.solver == 'LKH':
                tasks = [sp.get_lkh_args(max_trials=args.n_lkh_trials, init=args.init_tour) for sp in subps]
                results = multithread(lambda args: run_lkh(*args), tasks, cpus=n_cpus, show_bar=show_bar)
            elif args.solver == 'HGS':
                tasks = [sp.get_hgs_args(time_threshold=args.time_threshold) for sp in subps]
                results = multithread(lambda args: run_hgs(*args, seed=random.randrange(sys.maxsize) if not hasattr(args, 'seed') else args.seed), tasks, cpus=n_cpus, show_bar=show_bar)
            new_problems = []
            for p, subp, (new_tour, duration) in zip(ps, subps, results):
                subproblem_cache[subp] = (len(lkh_solutions), new_tour)
                subp.set_routes(unpack_routes(new_tour))
                actions.append((p.id, subp.action, len(lkh_solutions)))
                lkh_solutions.append((subp.total_dist, new_tour))
                new_problems.append((p.total_dist + subp.change_dist, p, subp))
            best_problems = sorted(new_problems, key=lambda x: x[0])

        if hasattr(args, 'improve_threshold'):
            best_problems = [x for x in best_problems if x[0] - fullps[-1].total_dist <= -args.improve_threshold]
        times.append(time() - start_time)
        if len(best_problems) == 0: continue # no better solution found, keep same problems
        best_problems = sorted(best_problems, key=lambda x: x[0])[:args.beam_width]

        fullps = [p.apply_subproblem(subp).set_prev(p.id, subp.action) for _, p, subp in best_problems]
        for fullp in fullps:
            problem_id += 1
            fullp.set_id(problem_id)
            transitions.append((fullp.prev_id, fullp.prev_action, fullp.id))
    return subproblem_cache, transitions, actions, lkh_solutions, times

def save_beam_search(save_path, *beam_args, kwargs_fn=None, **kwargs):
    nodes, demands, capacity, init_routes, args = beam_args
    subproblem_cache, transitions, actions, lkh_solutions, times = beam_search(*beam_args, **kwargs)

    lkh_dists, lkh_subtours = zip(*lkh_solutions)
    np.savez(save_path,
        xys=nodes,
        demands=demands,
        capacity=capacity,
        routes=pack_routes(init_routes),
        transitions=np.array(transitions, dtype=np.uint16),
        actions=np.array(actions, dtype=np.uint16),
        lkh_dists=np.array(lkh_dists, dtype=np.float32),
        lkh_routes=pad_each(lkh_subtours).astype(np.uint16),
        times=np.array(times),
        **kwargs.get('pkwargs', {}),
        **(kwargs_fn() if kwargs_fn else {}),
    )

def reconstruct_data(nodes, demands, capacity, init_routes, transitions, actions, lkh_dists, lkh_routes, ptype='CVRP', pkwargs={}, n_route_neighbors=15, generate_depth=30, n_subproblems=None, subproblem_temperature=0.0):
    """
    transitions: Tuples of (problem_id, action, next_problem_id)
    actions: Tuples of (problem_id, action, LKH solution index)
    lkh_solutions: Tuples of (routes, dist)
    """
    pkwargs = {k: pkwargs[k] for k in dict(CVRP=[], CVRPTW=['window', 'service_time', 'window_distance_scale'], VRPMPD=['is_pickup'])[ptype]}
    # If same (p_id, a) maps to multiple lkh_i (as in double LKH), the later one is kept
    pa2lkh = {(p_id, a): lkh_i for p_id, a, lkh_i in actions}
    p2alkh = defaultdict(list)
    for (p_id, a), lkh_i in pa2lkh.items():
        p2alkh[p_id].append((a, lkh_i))

    problem_id = 0
    problems = [VRFullProblem(nodes, demands, capacity, init_routes, ptype=ptype, pkwargs=pkwargs).set_id(problem_id).set_depth(0)]

    for problem_id, action, next_problem_id in transitions:
        p = problems[problem_id]
        if p is None or p.depth == generate_depth:
            problems.append(None)
            continue
        subp = p.get_subproblem(action, n_subproblems=n_subproblems, do_init_routes=False, n_route_neighbors=n_route_neighbors, temperature=subproblem_temperature)
        subp.set_routes(unpack_routes(lkh_routes[pa2lkh[p.id, action]]))
        nextp = p.apply_subproblem(subp).set_prev(p.id, action).set_id(next_problem_id).set_depth(p.depth + 1)
        nextp.delta = nextp.total_dist - p.total_dist
        problems.append(nextp)

    problems = [p for p in problems if p is not None]
    for p in problems: # Generate labels and unique_mask
        if p.id not in p2alkh: continue
        acts, lkh_idxs = np.array(sorted(p2alkh[p.id])).T
        _, unique_idxs = np.unique(lkh_idxs, return_index=True)
        p.unique_mask = np.zeros_like(acts, dtype=np.bool)
        p.unique_mask[unique_idxs] = True
        subp_dists = [p.route_dists[p.get_cluster(a, n_subproblems=n_subproblems, route_only=True, n_route_neighbors=n_route_neighbors, temperature=subproblem_temperature)].sum() for a in acts]
        p.lkh_idxs = lkh_idxs
        p.labels = lkh_dists[lkh_idxs] - subp_dists
        if len(acts) < len(p.routes):
            p.labels = dict(zip(acts, p.labels))
            p.lkh_idxs = dict(zip(acts, p.lkh_idxs))
    return problems

class ActionCallback:
    def __init__(self, args):
        self.args = args
        self.prev_subps = set()
        self.last_p = None
        self.n_repeats = []
        self.order = None

    def action_fn(self, p):
        args = self.args
        n_repeat = (self.n_repeats[-1] + 1) if p is self.last_p else 0
        self.n_repeats.append(n_repeat)

        if n_repeat:
            order = self.order
        else:
            self.order = order = self.action_order(p)

        if n_repeat >= len(order):
            return []
        action = order[n_repeat]
        while args.detect_duplicate and p.get_subproblem(action, n_subproblems=args.n_subproblems, n_route_neighbors=args.n_route_neighbors, temperature=args.subproblem_temperature) in self.prev_subps: # Make sure that the subproblem wasn't previously chosen
            n_repeat += 1
            if n_repeat >= len(order):
                return []
            action = order[n_repeat]
        actions = order[n_repeat: n_repeat + args.beam_width]
        self.last_p = p
        return actions

    def feedback_fn(self, actions, subproblems, feedbacks):
        if self.args.detect_duplicate:
            for action, subp, delta in zip(actions, subproblems, feedbacks):
                self.prev_subps.add(subp)

    def kwargs_fn(self):
        return dict(repeats=np.array(self.n_repeats, dtype=np.uint16))