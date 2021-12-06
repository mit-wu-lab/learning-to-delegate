# Learning to Delegate for Large-scale Vehicle Routing
This directory contains the code, data, and model for our NeurIPS 2021 Spotlight paper *Learning to Delegate for Large-scale Vehicle Routing*, which applies subproblem selection to large-scale vehicle routing problems (VRPs).

## Relevant Links
You may find this project at: [arXiv](https://arxiv.org/abs/2107.04139), [Project Website](https://mit-wu-lab.github.io/learning-to-delegate/), [OpenReview](https://openreview.net/forum?id=rm0I5y2zkG8), [Poster](https://github.com/mit-wu-lab/learning-to-delegate/blob/gh-pages/img/poster.png).
```
@inproceedings{li2021learning,
title={Learning to delegate for large-scale vehicle routing},
author={Sirui Li and Zhongxia Yan and Cathy Wu},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021}
}
```

## Code Instruction
Included in this directory are code and instructions to generate CVRP, CVRPTW, and VRPMPD problem instances with LKH-3 and HGS subsolvers (when applicable), run baselines mentioned in our paper, and train and evaluate subproblem regression and classification models.

We include model configurations and pretrained models for all experiments from our paper, which can be applied on given problem instances. We additionally include generated data for training all of these models.

## Overview
To apply our method and baseline methods on a given VRP distribution, we take the following steps (described in full details in our paper)

1. `generate_[initial,real_world].py`: [Generating Problem Instances](#generating-problem-instances). This may need to be repeated several times to generate training, validation, and/or test sets.
2. Training
    1. `generate_multiprocess.py`: [Generating Training and Validation Data](#generating-training-and-validation-data). We collect training trajectories by enumerating over all the subproblems at each step with a subsolver. We only collect training data for VRP distributions that we train on.
    2. [Preprocessing Training and Validation Data](#preprocessing-training-and-validation-data)
        1. `preprocess[_subproblems,].py`: Preprocess training data. We aggregate training data into a single file with more accessible formats for training. `preprocess_subproblems.py` preprocesses for regression, while `preprocess.py` preprocesses for classification.
        2. `concat_preprocessed.py`: If needed, we concatenate multiple preprocessing outputs into a single file to merge datasets for training.
    4. `supervised.py`: Training [Regression](#regression) or [Classification](#classification) subproblem selection model. We train on our model on the training data while tracking loss on the validation set. The VRP distributions that we train models for are discussed in our paper.
5. `supervised.py`: [Generating Solution Trajectories](#generating-solution-trajectories). This runs our iterative framework with a trained subproblem selection model (regression or classification).
6. Generating baseline solutions
    1. `generate_multiprocess.py`: [Subproblem Selection Baselines](#subproblem-selection-baselines).
    2. `run_[lkh,hgs].py`: [LKH-3](#lkh-3-baseline) or [HGS](#hgs-baseline) Baseline. Obtaining solutions using only the subsolver (i.e. only LKH-3 or HGS) without subproblem selection.

The commands that we give below creates the directory structure as shown in [Directory Structure](#directory-structure). If present, the arguments `n_cpus`, `n_process`, and/or `n_threads_per_process` control how much computational resources should be used. Note that for the final paper we measure runtime for each instance on a **single thread**.
1. Set `n_threads_per_process` to `1` when running `generate_initial.py` or `generate_real_world.py`
2. Do not use the `n_threads_per_process` argument when generating solution trajectories with subproblem selection baselines (`generate_multiprocess.py`). This will let `n_threads_per_process` be `1`.

We give a rough estimate of the single-CPU computation time (except for training) of each process under the Uniform CVRP section. In practice we leverage parallelism to run multiple instances in parallel.

### Directory Structure
We list below the directory structure of this repo and a [zip file](https://www.dropbox.com/s/37n002kezc3t3w0/learning-to-delegate.zip?dl=0), which contains all training / validation data and model configurations / checkpoints. After unzipping the zip file, you may need to manually move `generations/` and `exps/` so they are under this repo cloned from Github. The zip file is 10Gb in size with many small files and may take 30 minutes to an hour to unzip.

```
learning-to-delegate/  # Top directory of this Github repo
 ├─ README.md  # This markdown file
 ├─ *.py  # Code files
 ├─ lkh3/  # LKH-3 solver
 ├─ hgs/  # HGS solver
 ├─ VRP_Instances_Belgium/  # Real-world dataset from https://antor.uantwerpen.be/xxlrouting/
 ├─ example.ipynb  # Example script for computing speedup and plotting our paper's curves
 ├─ generations/  # Zipped problem instances and solution trajectories
 │   ├─ uniform_N[500,1000,2000,3000]/
 │   │   ├─ problems_[train,val,test].npz  # Problem instances with initializations
 │   │   ├─ subproblem_selection_lkh/  # Data from LKH-based subproblem selection
 │   │   │   ├─ train_routeneighbors10/  # k = 10
 │   │   │   │   └─ [0-1999].npz
 │   │   │   ├─ val_routeneighbors10/
 │   │   │   │   └─ [0-39].npz
 │   │   │   └─ [train,val]_routeneighbors5*/  # k = 5
 │   │   └─ subproblem_selection_hgs/
 │   │       └─ [train,val]_routeneighbors10*/
 │   ├─ [clustered,mixed]_nc[3,5,7]_N[500,1000,2000]/
 │   │   ├─ problems_[train,val,test].npz
 │   │   ├─ subproblem_selection_lkh/
 │   │   │   ├─ train_routeneighbors10*/
 │   │   │   │   └─ [0-499].npz
 │   │   │   └─ val_routeneighbors10*/
 │   │   │       └─ [0-9].npz
 │   │   └─ subproblem_selection_hgs/
 │   │       └─ [train,val]_routeneighbors10*/
 │   ├─ real_N2000/  # Problem instances in the real-world CVRP distribution
 │   │   └─ problems_[val,test].npz
 │   └─ [cvrptw,vrpmpd]_uniform_N500/
 │       ├─ problems_[train,val,test].npz
 │       └─ subproblem_selection_lkh/
 │           ├─ train_routeneighbors5*/
 │           │   └─ [0-1999].npz
 │           └─ val_routeneighbors5*/
 │               └─ [0-39].npz
 └─ exps/  # Zipped trained models
     └─ [,cvrptw_,vrpmpd_][uniform,clustered]_[merge,N500,N1000,N2000]_routeneighbors[5,10]*/
         └─ <exp_name>/
             ├─ config.yaml  # Model parameters
             ├─ events.out.tfevents.*  # Tensorboard log
             └─ models/  # Model checkpoints
                 └─ [0,10000,20000,30000,40000].pth
```

When using problem instances and training data from the zip, you should skip [Generating Problem Instances](#generating-problem-instances) and [Generating Training and Validation Data](#generating-training-and-validation-data) (which is the most computationally intensive step).

**Note that problem instances in the zip file contain initialization times specific to our machines**, which are relatively insignificant overall. If you'd like to compute initialization times on your own machines, you may want to generate new `test` set instances.

If only using trained model checkpoints from the zip file, you can skip straight to [Generating Solution Trajectories](#generating-solution-trajectories) and baselines ([Subproblem Selection Baselines](#subproblem-selection-baselines), [LKH-3 Baseline](#lkh-3-baseline), or [HGS Baseline](#hgs-baseline)).


The zip file also contains example solution trajectories of our iterative framework and baselines for `N = 2000` uniform CVRP, comparing our method with the Random and LKH-3 baselines from the paper, as detailed in [Example Analysis and Plotting](#example-analysis-and-plotting).

## Environment Setup
We implement our models with PyTorch in Python 3.8. We include setup instructions here.

```
conda create -n vrp python=3.8
conda activate vrp
conda install numpy scipy
pip install pyyaml tensorboard
pip install scikit-learn # Architecture ablations only

# If you would like to run example.ipynb
conda install pandas matplotlib
conda install -c conda-forge notebook
pip install ipykernel
python -m ipykernel install --user --name vrp
# Make sure to select the vrp kernel from jupyter
```

Refer to the [PyTorch website](https://pytorch.org/) to install PyTorch 1.8 with GPU if possible.

Refer to the [PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install PyTorch Geometric to be compatible with the PyTorch and CUDA version (we use PyTorch 1.8 with CUDA 10.2). PyTorch Geometric is only used for the classification models and can be commented out when using regression only.

Follow the instructions in `lkh3/LKH-3.0.4/README.txt` to compile [LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3).

Follow the instructions in `hgs/HGS-CVRP/README.md` to compile [HGS](https://github.com/vidalt/HGS-CVRP).

## Uniform CVRP
### Generating Problem Instances
Given a problem instance size 500, 1000, 2000, or 3000 (or any other integer), these commands generates 2000 instances for `train` set, 40 instances for `val` set, and 40 for `test` set. `SAVE_DIR` can be set arbitrarily.

This should be relatively quick (less than one minute) per instance.

```
export SPLIT=val # options: [train,val,test]
export N=500 # options: [500,1000,2000,3000]
export SAVE_DIR=generations/uniform_N$N
export N_INSTANCES=40 # 2000 for train, 40 for val and test. May set to less to save time
export N_CPUS=40 # set this according to your compute budget

python generate_initial.py $SAVE_DIR $SPLIT $N --n_instances $N_INSTANCES --n_process $N_CPUS --n_threads_per_process 1
```

### LKH-3 Baseline
We run 30000 LKH steps for `N = 500, 1000, 2000` and 50000 LKH steps for `N = 3000`. For our experiments we use 5 runs per instance, but one can run over fewer runs per instance  (e.g. 1) to save time.

This should take several hours to run 30000 LKH steps per instance. Note that LKH may terminate early before all `LKH_STEPS` have been run.

```
export SPLIT=val # options: [val,test]
export LKH_STEPS=30000 # use 50000 for N = 3000
export N_RUNS=5 # use 1 for experimentation to save time

python run_lkh.py $SAVE_DIR/lkh $SPLIT --save_dir $SAVE_DIR --n_lkh_trials $LKH_STEPS --init_tour --index_start 0 --index_end $N_INSTANCES --n_runs $N_RUNS --n_cpus $N_CPUS
```

### Subproblem Selection Baselines
As described in our paper, these are the Random, Min Count, Max Min Dist baseline. Each baseline should take ~10 minutes to over an hour per instance, depending on the `N`, `K`, and `DEPTH`. Note that generation may terminate early before `DEPTH` iterations have been run.
```
export SPLIT=val # options: [val,test]
export METHOD=sample # options: use [sample,min_count,max_min_dist] for [Random, Min Count, Max Min Dist] respectively
export K=10 # options: [5,10]
export DEPTH=400 # respectively for N = [500,1000,2000,3000], use [400,600,1200,2000] for K = 10 or [1000,2000,3000,4500] for K = 5
export N_INSTANCES=40
export N_RUNS=5 # use 1 for experimentation to save time
export DATASET_DIR=$SAVE_DIR/subproblem_selection_lkh

MKL_NUM_THREADS=1 python generate_multiprocess.py $DATASET_DIR $SPLIT --save_dir $SAVE_DIR --n_lkh_trials 500 --n_cpus $N_CPUS --n_runs $N_RUNS --index_start 0 --index_end $N_INSTANCES --beam_width 1 --$METHOD --n_route_neighbors $K --generate_depth $DEPTH
```

The solutions trajectories are saved in a particular format; please see [Example Analysis and Plotting](#example-analysis-and-plotting) for how to unpack the trajectories.

### Generating Training and Validation Data
This should take 30 mins to an hour **per instance** depending on the `N`, `K`, and `DEPTH`. We only generate data for `N = [500,1000]` for `K = 10` or `N = [500,1000,2000]` for `K = 5`.
```
export SPLIT=train # options: [train,val] for training and validation respectively
export N_INSTANCES=2000 # use 2000 for train and 40 for val
export K=10 # options: [5,10]
export DEPTH=30 # for K = 10: use 30 for N = [500,1000]; for K = 5: use [40,80,160] for N = [500,1000,2000]
export DATASET_DIR=$SAVE_DIR/subproblem_selection_lkh

python generate_multiprocess.py $DATASET_DIR $SPLIT --save_dir $SAVE_DIR --n_lkh_trials 500 --n_cpus $N_CPUS --index_start 0 --index_end $N_INSTANCES --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH
```

### Regression
#### Preprocessing Training and Validation Data
This should be very quick to run (less than 1 minute).

```
export K=10 # options: [5,10]
export DEPTH=30 # This should be the same as the depth used in Generating Training and Validation Data

python preprocess_subproblems.py $DATASET_DIR val train --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH --n_cpus $N_CPUS
```

Preprocessing should save some npz files, which can be used for training and validation. In our regression experiments (but not our classification experiments) we jointly train on preprocessed data from multiple problem sizes. Therefore we run `concat_preprocessed.py` to concatenate the training datas from different sizes, where `PATH1`, `PATH2`, ... are the paths to the saved npz files (whose names are printed by the `preprocess_subproblems.py` command above).

We run `concat_preprocessed.py` again for paths to validation data. Therefore, we should obtain a single training npz file and a single validation npz file.

```
export SPLIT=train # Need to run for both train and val
export MERGED_SAVE_DIR=generations/uniform_merge
export PATH1=
export PATH2=
export DATA_SUFFIX= # This can be be used to distinguish between multiple merged training sets
export MERGED_DATASET_DIR=$MERGED_SAVE_DIR/subproblem_selection_lkh
export OUT_PATH=$MERGED_DATASET_DIR/$SPLIT${DATA_SUFFIX}_subproblems.npz

python concat_preprocessed.py $PATH1 $PATH2 $OUT_PATH
```

#### Training
This should take around 6 hours on a NVIDIA V100 GPU. Here we provide the command with our best hyperparameters.

```
export K=10 # options: [5,10], this should be the same as the generated data
export TRAIN_DIR=exps/regression_model # The name can be arbitrary
export TRAIN_STEPS=40000 # The validation loss at step 40000 is always the best

python supervised.py $MERGED_DATASET_DIR $TRAIN_DIR --data_suffix $DATA_SUFFIX --fit_subproblem --augment_rotate --augment_flip --lr 0.001 --n_batch 2048 --n_layers 6 --transformer_heads 8 --n_route_neighbors $K --n_steps $TRAIN_STEPS
```

### Classification
#### Preprocessing Training and Validation Data
This should be very quick to run (less than 1 minute).
```
export K=10 # options: [5,10]
export DEPTH=30 # This should be the same as the depth used in Generating Training and Validation Data

python preprocess.py $DATASET_DIR val train --beam_width 1 --n_route_neighbors $K --generate_depth $DEPTH --n_cpus $N_CPUS
```

Preprocessing should save some npz files, which can be used for training and validation.

#### Training
This should take around 6 hours, 12 hours, or 24 hours on a NVIDIA V100 GPU for `N = 500, 1000, 2000` respectively.

One must set the `DATA_SUFFIX` variable so that the training and validation data generated above is at `$SAVE_DIR/subproblem_selection_lkh/$SPLIT${DATA_SUFFIX}.npz`.
```
export DATA_SUFFIX= # Set this variable according to instructions
export TRAIN_DIR=exps/classification_model # This can be set arbitrarily
export PERTURB_NODE=0.05 # options: [0.05,0.01] for N = [500,1000] respectively
export PERTURB_ROUTE=0.005 # options: [0.005,0.001] for N = [500,1000] respectively
export K=10 # this should be the same as the generated data
export TRAIN_STEPS=40000

python supervised.py $DATASET_DIR $TRAIN_DIR --data_suffix $DATA_SUFFIX --augment_rotate --augment_flip --augment_perturb_node $PERTURB_NODE --augment_perturb_route $PERTURB_ROUTE --lr 0.001 --n_route_neighbors $K --use_layer_norm --use_x_fc --n_batch 256 --n_steps $TRAIN_STEPS
```

### Generating Solution Trajectories
This takes anywhere from ~10 minutes to over an hour per instance, depending on the `N`, `K`, and `DEPTH`. Note that generation may terminate early before `DEPTH` iterations have been run.

When we evaluate trained models on a particular problem size `N`, we set the `GENERATE_SAVE_DIR` to correspond to the `SAVE_DIR` for that particular `N`. Similarly, if we would like to run the model on other data (e.g. clustered data), we change `GENERATE_SAVE_DIR` to be the `SAVE_DIR` of the target data.
```
export TRAIN_DIR=exps/regression_model # Directory for regression or classification model
export GENERATE_CHECKPOINT_STEP=40000

export GENERATE_SAVE_DIR=$SAVE_DIR # This should be set as described above
export GENERATE_PARTITION=val # options: [val,test]
export GENERATE_SUFFIX=_val # A suffix which helps distinguish between different $GENERATE_SAVE_DIR

export DEPTH=400 # respectively for N = [500,1000,2000,3000], use [400,600,1200,2000] for K = 10 or [1000,2000,3000,4500] for K = 5
export N_INSTANCES=40
export N_RUNS=5 # use 1 for experimentation to save time

MKL_NUM_THREADS=1 python supervised.py $DATASET_DIR $TRAIN_DIR --generate --step $GENERATE_CHECKPOINT_STEP --generate_partition $GENERATE_PARTITION --save_dir $GENERATE_SAVE_DIR --save_suffix $GENERATE_SUFFIX --generate_depth $DEPTH --generate_index_start 0 --generate_index_end $N_INSTANCES --n_lkh_trials 500 --n_trajectories $N_RUNS --n_cpus $N_CPUS --device cpu
```
The solutions trajectories are saved in a particular format; please see [Example Analysis and Plotting](#example-analysis-and-plotting) for how to unpack the trajectories.

## Clustered and Mixed CVRP
For clustered and mixed CVRP distributions, only the problem generation differs from uniform CVRP.
### Generating Problem Instances
Given a problem instance size `N = [500,1000,2000]`, `NC = [3,5,7]` cluster centers, and whether we want clustered or mixed CVRP distributions, these commands generates 500 instances for `train` set, 10 instances for `val` set, and 10 for `test` set. Note that we do not train on `N = 2000` data, so there's no need to generate training instances for `N = 2000`. `SAVE_DIR` can be set arbitrarily.

This should be relatively quick (less than one minute) per instance.

```
export SPLIT=val # options: [train,val,test]
export NC=3 # options: [3,5,7]
export N=500 # options: [500,1000,2000,3000]
export SAVE_DIR_CLUSTERED=generations/clustered_nc${NC}_N$N
export SAVE_DIR_MIXED=generations/mixed_nc${NC}_N$N
export N_INSTANCES=10 # options: 10 if SPLIT = [val,test]; 500 if SPLIT = train
export N_CPUS=40 # set this according to your compute budget

# Clustered
python generate_initial.py $SAVE_DIR_CLUSTERED $SPLIT $N --n_c $NC --n_instances $N_INSTANCES --n_process $N_CPUS --n_threads_per_process 1

# Mixed
python generate_initial.py $SAVE_DIR_MIXED $SPLIT $N --n_c $NC --mixed --n_instances $N_INSTANCES --n_process $N_CPUS --n_threads_per_process 1
```

## Real-world CVRP
For the real-world CVRP distribution, only the problem generation differs from uniform CVRP.
### Generating Problem Instances
We generate our real-world distribution from real-world CVRP instances found in the `VRP_Instances_Belgium` directory. We only need to generate `N = 2000` instances for the `val` and `test` sets, as we do not train on this distribution.

```
export REAL_DIR=VRP_Instances_Belgium
export SAVE_DIR=generations/real_N2000
export SPLIT=val # options: [val,test]
export N=2000
export N_INSTANCES_PER_EXAMPLE=5
export N_CPUS=40 # set this according to your compute budget

python generate_real_world.py $REAL_DIR $SAVE_DIR $SPLIT $N --n_instances_per_example $N_INSTANCES_PER_EXAMPLE --n_process $N_CPUS --n_threads_per_process 1
```

## HGS Subsolver
We provide instructions on using HGS as the subsolver instead of LKH-3. Unlike LKH-3, HGS does not take an initial solution and is run for a particular amount of time rather than a particular number of steps.
### HGS Baseline
For uniform distribution, we run HGS for `T = 1800, 4620, 8940, 30000` seconds for `N = 500, 1000, 2000, 3000` respectively. For clustered or mixed distribution, we run HGS for `T = 2000, 5000, 10000` seconds for `N = 500, 1000, 2000` respectively. For our experiments we use 5 runs per instance, but one can run over fewer runs per instance (e.g. 1) to save time. This should take the same amount of time as the corresponding LKH-3 runs.

```
export SAVE_DIR=generations/uniform_N500 # This could be clustered, mixed, or real as well
export T=1800 # Set this according to the instructions given
export SPLIT=val # options: [val,test]
export N_INSTANCES=40 # 2000 for train, 40 for val and test. May set to less to save time
export N_RUNS=5 # use 1 for experimentation to save time

python run_hgs.py $SAVE_DIR/hgs $SPLIT --save_dir $SAVE_DIR --time_threshold $T --index_start 0 --index_end $N_INSTANCES --n_runs $N_RUNS --n_cpus $N_CPUS
```

### Subproblem Selection Baselines
This is very similar to the [Subproblem Selection Baselines](#subproblem-selection-baselines) commands for LKH-3. We additionally set the following environmental variables.
```
export DATASET_DIR=$SAVE_DIR/subproblem_selection_hgs
export T=1 # We run HGS for 1 second on each subproblem
export K=10
export DEPTH=2000 # use [2000,4500,15000,40000] respectively for N = [500,1000,2000,3000]
```
We add the following flags to the previous command: `--solver HGS --time_threshold $T`.

### Training
The training data generation process and training process are identical to the LKH-3 based [Training](#training) above, so we do not repeat commands here.

### Generating Solution Trajectories
This is very similar to the command for [Generating Solution Trajectories](#generating-solution-trajectories) with the LKH-3 subsolver. We additionally set the following environmental variables.
```
export T=1 # We run HGS for 1 second on each subproblem
export DEPTH=2000 # use [2000,4500,15000,40000] respectively for N = [500,1000,2000,3000]
```
We add the following flags to the previous command: `--solver HGS --time_threshold $T`.

## CVRPTW
For the CVRPTW distribution, the main difference from uniform CVRP is in problem generation.

For other steps of the framework, add `--ptype CVRPTW` as an argument to every uniform CVRP command above. Note that HGS does not handle CVRPTW problem instances.
### Generating Problem Instances
```
export SPLIT=val # options: [train,val,test]
export N=500 # options: [500,1000,2000,3000]
export SAVE_DIR=generations/cvrptw_uniform_N$N
export N_CPUS=40 # set this according to your compute budget
export N_INSTANCES=40 # 2000 for train, 40 for val and test. May set to less to save time

python generate_initial.py $SAVE_DIR $SPLIT $N --ptype CVRPTW --service_time 0.2 --max_window_width 1.0 --n_instances $N_INSTANCES --n_process $N_CPUS --n_threads_per_process 1
```

## VRPMPD
For the CVRPTW distribution, the main difference from uniform CVRP is in problem generation.

For other steps of the framework, add `--ptype VRPMPD` as an argument to every uniform CVRP command above. Note that HGS does not handle CVRPTW problem instances.

### Generating Problem Instances
We use a capacity of `25` instead of `50` for VRPMPD as this keeps route lengths around the same as CVRP.
```
export SPLIT=val # options: [train,val,test]
export N=500 # options: [500,1000,2000,3000]
export SAVE_DIR=generations/vrpmpd_uniform_N$N
export N_CPUS=40 # set this according to your compute budget
export N_INSTANCES=40 # 2000 for train, 40 for val and test. May set to less to save time

python generate_initial.py $SAVE_DIR $SPLIT $N --ptype VRPMPD --capacity 25 --n_instances $N_INSTANCES --n_process $N_CPUS --n_threads_per_process 1
```

## Other Ablations
We include commands for other ablations described in our appendix.

### Initialization Ablation
We must generate the initial problems with the default initialization first, before generating alternative initial solutions for the same problems. Therefore, **after running the `generate_initial.py` command in [Generating Problem Instances](#generating-problem-instances)**, we run the **`generate_initial.py` command** again with additional flag `--naive_init` for the `L = 0` initialization method in the paper or `--n_lkh_trials $L` for initialization methods running `L` LKH steps on each initialization partition.

For generating solution trajectories for [Subproblem Selection Baselines](#subproblem-selection-baselines) (`generate_multiprocess.py`), we add an additional flag `--partition_suffix initnaive` or `--partition_suffix lkh$L`, respectively. For [Generating Solution Trajectories](#generating-solution-trajectories) with our trained model (`supervised.py`), we run the same command with additional flag `--generate_partition_suffix initnaive` or `--generate_partition_suffix lkh$L`.

### Architecture Ablation
#### FCNN
For this ablation, we use the same preprocessed training and validation data as our regression method, therefore the only change adding the `--fc_only` flag to the training command, which replaces the Transformer with a fully connected neural network.

#### Linear, MLP, RandomForest
For these ablation, the input format consists of 33 features representing the statistics of each subproblem, so we need to use a different preprocessing for the training and validation data by running the `preprocess_subproblems.py` command from [Preprocessing Training and Validation Data](#preprocessing-training-and-validation-data) with the additional `--statistics` flag.

Similarly, we run the `concat_preprocessed.py` command (remember to change `PATH1`, `PATH2`, and `OUT_PATH` accordingly to end with `_subproblem_statistics.npz` instead of `_subproblems.npz`) with the additional `--statistics` flag.

Finally, for [Training](#training) we run the `supervised.py` command with the additional arguments `--fit_statistics --use_sklearn --sklearn_parameters $SKLEARN_PARAMS`, where `SKLEARN_PARAMS` is expressed in YAML format. In particular, we use `SKLEARN_PARAMS="model: ElasticNet\nalpha: 0"` for Linear or `SKLEARN_PARAMS="model: MLPRegressor\nalpha: 0"` for MLP. **We do not recommend training with a RandomForest-based model**, which achieves similar validation MSE as Linear or MLP but takes up excessive disk space (e.g. 30Gb) when stored and is too memory intensive to execute for generating solution trajectories in parallel.

## Example Analysis and Plotting
We provide the `example.ipynb` script for computing speedup and plotting our paper results for the example solution trajectories provided by our zip file. Besides these examples, we do not provide other solution trajectories as solution times are benchmarked using our own servers, and other servers are likely to see different solutions times. Solution trajectories can be generated using the code in this repo and our trained models from the zip file.

Beyond demonstrating how to unpack the solution trajectory files, we do not elaborate more on the file format. Please refer to the code for full details.
