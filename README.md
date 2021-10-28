# Learning to Delegate for Large-scale Vehicle Routing
We provide clean and extensible code for subproblem selection in large-scale vehicle problems (VRPs). Included in this directory are code and instructions to generate CVRP, CVRPTW, and VRPMPD problem instances, run baselines mentioned in our paper, and train and evaluate subproblem regression and subproblem classification models. We also include a pretrained model which can be applied on given problem instances.

## Environment Setup
Our models are trained with PyTorch in Python 3.8. We include setup instructions here.

```
conda create -n vrp python=3.8
conda activate vrp
conda install numpy scipy
pip install pyyaml tensorboard
```

Refer to the [PyTorch website](https://pytorch.org/) to install PyTorch 1.8 with GPU if possible.

Refer to the [PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install PyTorch Geometric to be compatible with the PyTorch and CUDA version (we use PyTorch 1.8 with CUDA 10.2).

Follow the instructions in `lkh3/LKH-3.0.4/README.txt` to compile [LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3/).

## Overview
To apply our method on a given VRP distribution, we take the following steps

1. Generate the problem instances. This may need to be repeated several times to generate training, validation, and/or test sets.
2. Generating training data. This is optional if training is not needed on the particular VRP distribution
3. Training. This is optional if training is not needed on the particular VRP distribution.
4. Evaluating trained models on validation or test set. This runs our iterative framework with a trained subproblem selection model (regression or classification).
5. Evaluating baselines on validation or test set. We provide commands to run our LKH-3 baseline and subproblem selection baselines Random, Min Count, and Max Min Dist.

Below we provide the specific commands that we run for each VRP distribution. Note that in our code, we use the `test` set to refer to the validation set on which we select hyperparameters and the `finaltest` set to refer to the test set on which we report final results with the frozen hyperparameters. Similarly, for real-world CVRP instances, we use the `real` set to refer to the validation set and the `finalreal` set to refer to the test set.

If present, the arguments `n_cpus`, `n_process`, and/or `n_threads_per_process` control how much computational resources should be used.

We give a rough estimate of the single-CPU computation time (except for training) of each process under the Uniform CVRP section. In practice we leverage parallelism to run multiple instances in parallel.

## Uniform CVRP
### Generating Problem Instance
Given a problem instance size 500, 1000, 2000, or 3000 (or any other integer), these commands generates 2000 instances for `train` set, 40 instances for `test` set, and 40 for `finaltest` set. `SAVE_DIR` can be set arbitrarily.

This should be relatively quick (less than one minute) per instance.

```
export SAVE_DIR=save
export SPLIT=test # options: [train,test,finaltest]
export N=500 # options: [500,1000,2000,3000]

python generate_initial.py $SAVE_DIR $SPLIT $N --n_process 40 --n_threads_per_process 1
```

### LKH-3 Baseline
We run 30000 LKH steps for `N = 500, 1000, 2000` and 50000 LKH steps for `N = 3000`. For our experiments we use 5 runs per instance, but one can run over fewer runs per instance  (e.g. 1) to save time.

This should take several hours to run 30000 LKH steps per instance.

```
export SPLIT=test # options: [test,finaltest]
export LKH_STEPS=30000 # use 50000 for N = 3000
export NUM_INSTANCES=40
export N_RUNS=5 # use 1 for experimentation to save time

python run_lkh.py $SAVE_DIR/fulllkhinit30000 $SPLIT --save_dir $SAVE_DIR --n_lkh_trials $LKH_STEPS --n_cpus 40 --index_start 0 --index_end $NUM_INSTANCES --n_runs $N_RUNS --init_tour
```

### Subproblem Selection Baselines: Random, Min Count, Max Min Dist
This should take ~10 minutes to over an hour per instance, depending on the `N`, `K`, and `DEPTH`.
```
export SPLIT=test # options: [test,finaltest]
export NUM_INSTANCES=40
export METHOD=sample # options: use [sample,min_count,max_min_dist] for [Random, Min Count, Max Min Dist] respectively
export K=10 # options: [5,10]
export DEPTH=400 # respectively for N = [500,1000,2000,3000], use [400,600,1200,2000] for K = 10 or [1000,2000,3000,4500] for K = 5
export N_RUNS=5 # use 1 for experimentation to save time

MKL_NUM_THREADS=1 python generate_multiprocess.py $SAVE_DIR/generations $SPLIT --save_dir $SAVE_DIR --n_lkh_trials 500 --n_cpus 40 --n_runs $N_RUNS --index_start 0 --index_end $NUM_INSTANCES --beam_width 1 --$METHOD --n_route_neighbors $K --generate_depth $DEPTH
```

### Generating Training and Validation Data
This should take 30 mins to an hour per instance depending on the `N`, `K`, and `DEPTH`. We only generate data for `N = [500,1000]` for `K = 10` or `N = [500,1000,2000]` for `K = 5`.
```
export SPLIT=train # options: [train,test] for training and validation respectively
export NUM_INSTANCES=2000 # use 2000 for SPLIT = train and 40 for SPLIT = test
export K=10 # options: [5,10]
export DEPTH=30 # for K = 10: use 30 for N = [500,1000]; for K = 5: use [40,80,160] for N = [500,1000,2000]

python generate_multiprocess.py $SAVE_DIR/generations $SPLIT --save_dir $SAVE_DIR --n_lkh_trials 500 --n_cpus 40 --index_start 0 --index_end $NUM_INSTANCES --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH
```

### Regression
#### Training and Validation Data Preprocessing
This should be very quick to run (less than 1 minute).

```
export K=10 # options: [5,10]
export DEPTH=30 # This should be the same as the depth used in Generating Training and Validation Data

python preprocess_subproblems.py $SAVE_DIR/generations test train --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH --n_cpus 40
```

Preprocessing should save some npz files, which can be used for training and validation. In our regression experiments (but not our classification experiments) we jointly train on preprocessed data from multiple problem sizes. Therefore we run the following to concatenate the training datas from different sizes, where `PATH1`, `PATH2`, ... are the paths to the saved npz files (whose names are printed by the `preprocess_subproblems.py` command above). We run the same command again for paths to validation data. Therefore, we should obtain a single training npz file and a single validation npz file.

```
export MERGED_SAVE_DIR=save_merge
export OUT_PATH=$MERGED_SAVE_DIR/generations/train.npz # replace train.npz with test.npz as necessary

python concat_preprocessed.py $PATH1 $PATH2 $OUT_PATH
```

#### Training
This should take around 6 hours on a NVIDIA V100 GPU. Here we provide the command with our best hyperparameters.

One must set the `DATA_SUFFIX` variable so that the training data generated above is at `$SAVE_DIR/generations/train${DATA_SUFFIX}_subproblems.npz` and validation data generated above is at `$SAVE_DIR/generations/test${DATA_SUFFIX}_subproblems.npz`.
```
export DATA_SAVE_DIR=$MERGED_SAVE_DIR
export MODEL_SAVE_DIR=exps/regression_model
export DATA_SUFFIX= # Set this variable according to instructions
export K=10 # options: [5,10], this should be the same as the generated data
export TRAIN_STEPS=40000

python supervised.py $DATA_SAVE_DIR/generations $MODEL_SAVE_DIR --data_suffix $DATA_SUFFIX --fit_subproblem --augment_rotate --augment_flip --lr 0.001 --n_batch 2048 --n_layers 6 --transformer_heads 8 --n_route_neighbors $K --n_steps $TRAIN_STEPS
```

### Classification
#### Training and Validation Data Preprocessing
This should be very quick to run (less than 1 minute).
```
export K=10 # options: [5,10]
export DEPTH=30 # This should be the same as the depth used in Generating Training and Validation Data

python preprocess.py $SAVE_DIR/generations test train --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH --n_cpus 40
```

Preprocessing should save some npz files, which can be used for training and validation.

#### Training
This should take around 6 hours, 12 hours, or 24 hours on a NVIDIA V100 GPU for `N = 500, 1000, 2000` respectively.

One must set the `DATA_SUFFIX` variable so that the training data generated above is at `$SAVE_DIR/generations/train${DATA_SUFFIX}.npz` and validation data generated above is at `$SAVE_DIR/generations/test${DATA_SUFFIX}.npz`.
```
export DATA_SAVE_DIR=$SAVE_DIR
export MODEL_SAVE_DIR=exps/classification_model
export DATA_SUFFIX= # Set this variable according to instructions
export PERTURB_NODE=0.05 # options: [0.05,0.01] for N = [500,1000] respectively
export PERTURB_ROUTE=0.005 # options: [0.005,0.001] for N = [500,1000] respectively
export K=10 # this should be the same as the generated data
export TRAIN_STEPS=40000

python supervised.py $DATA_SAVE_DIR/generations $MODEL_SAVE_DIR --data_suffix $DATA_SUFFIX --augment_rotate --augment_flip --augment_perturb_node $PERTURB_NODE --augment_perturb_route $PERTURB_ROUTE --lr 0.001 --n_route_neighbors $K --use_layer_norm --use_x_fc --n_batch 256 --n_steps $TRAIN_STEPS
```

### Evaluation
This should take ~10 minutes to over an hour per instance, depending on the `N`, `K`, and `DEPTH`.

When we evaluate trained models on a particular problem size `N`, we set the `GENERATE_SAVE_DIR` to correspond to the `SAVE_DIR` for that particular `N`. Similarly, if we would like to run the model on other data (e.g. clustered data), we change `GENERATE_SAVE_DIR` to be the `SAVE_DIR` of the target data.
```
export MODEL_SAVE_DIR=exps/regression_model # Directory for regression or classification model
export EVAL_STEP=40000
export EVAL_SPLIT=test # options: [test,finaltest]
export GENERATE_SAVE_DIR=$SAVE_DIR
export GENERATE_SUFFIX=_abcdef # A suffix which helps distinguish between different $GENERATE_SAVE_DIR
export DEPTH=400 # respectively for N = [500,1000,2000,3000], use [400,600,1200,2000] for K = 10 or [1000,2000,3000,4500] for K = 5
export N_RUNS=5 # use 1 for experimentation to save time

MKL_NUM_THREADS=1 python supervised.py $DATA_SAVE_DIR/generations $MODEL_SAVE_DIR --generate --step $EVAL_STEP --generate_partition $EVAL_SPLIT --save_dir $GENERATE_SAVE_DIR --save_suffix $GENERATE_SUFFIX --generate_depth $DEPTH --n_lkh_trials 500 --n_trajectories $N_RUNS --device cpu
```

## Clustered and Mixed CVRP
For clustered and mixed CVRP distributions, only the problem generation differs from uniform CVRP.
### Generating Problem Instance
Given a problem instance size `N = [500,1000,2000]`, `NC = [3,5,7]` cluster centers, and whether we want clustered or mixed CVRP distributions, these commands generates 500 instances for `train` set, 10 instances for `test` set, and 10 for `finaltest` set. Note that we do not train on `N = 2000` data, so there's no need to generate training instances for `N = 2000`. `SAVE_DIR` can be set arbitrarily.

This should be relatively quick (less than one minute) per instance.

```
export SAVE_DIR_CLUSTERED=save_clustered
export SAVE_DIR_MIXED=save_mixed
export SPLIT=test # options: [train,test,finaltest]
export N=500 # options: [500,1000,2000,3000]
export N_INSTANCES=10 # options: 10 if SPLIT = [test,finaltest]; 500 if SPLIT = train
export NC=3 # options: [3,5,7]

# Clustered
python generate_initial.py $SAVE_DIR_CLUSTERED $SPLIT $N --n_c $NC --n_instances $N_INSTANCES --n_process 40 --n_threads_per_process 1

# Mixed
python generate_initial.py $SAVE_DIR_MIXED $SPLIT $N --n_c $NC --mixed --n_instances $N_INSTANCES --n_process 40 --n_threads_per_process 1
```

## Real-world CVRP
For the real-world CVRP distribution, only the problem generation differs from uniform CVRP.
### Generating Problem Instance
We generate our real-world distribution from real-world CVRP instances found in the `VRP_Instances_Belgium` directory. We only need to generate `N = 2000` instances for the `test` and `finaltest` sets, as we do not train on this distribution.

```
export REAL_DIR=VRP_Instances_Belgium
export SAVE_DIR=save_real
export SPLIT=test # options: [test,finaltest]
export N=2000
export N_INSTANCES_PER_EXAMPLE=1
export NC=3 # options: [3,5,7]

python generate_real_world.py $REAL_DIR $SAVE_DIR $SPLIT $N --n_instances_per_example $N_INSTANCES_PER_EXAMPLE --n_process 40 --n_threads_per_process 1
```
## CVRPTW
For the CVRPTW distribution, the main difference from uniform CVRP is in problem generation. For other steps of the framework, add `--ptype CVRPTW` as an argument to every uniform CVRP command.
### Generating Problem Instance
```
export SAVE_DIR=save_cvrptw
export SPLIT=test # options: [train,test,finaltest]
export N=500 # options: [500,1000,2000,3000]

python generate_initial.py $SAVE_DIR $SPLIT $N --ptype CVRPTW --service_time 0.2 --max_window_width 1.0 --n_process 40 --n_threads_per_process 1
```

## VRPMPD
For the VRPMPD distribution, add `--ptype VRPMPD` as an argument to every uniform CVRP command.
