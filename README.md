# Code for "Training-Free Neural Active Learning with Initialization-Robustness Guarantees"

Code for paper titled "Training-Free Neural Active Learning with Initialization-Robustness Guarantees", accepted at ICML 2023.

## Running code

First, install requirements and package with `pip install -e .`

Then, to run active learning process, run the file `experiments/al.py`. To then do training on the selected points, run the file `experiments/train.py`.

Alternatively, you may run
```
./run_experiments_caller.sh [experiment number] [experiments mode] [which gpu id to use] [also do training after AL (optional)]
```
For example, 
```
./run_experiments_caller.sh exp35 211 0
```
which will run:
- Experiment code 35 (which calls on the relevant config files, see `experiments/experiments_cases.sh` for list of experiment settings), 
- With experiment setup code 211 (see `experiments/run_experiments_caller.sh` for all the settings, some are for running locally and some are written for running on Slurm machines),
- On GPU number 0.

To add your own config files, see `experiments/conf`.

To extend your own active learning methods, add class to `al_ntk/al_algorithms`, then register it on `al_ntk/experiments/al_utils/maps.py`.

To plot the relevant result graphs, use `experiments/plot_graphs.py`.

## Code source

Some of the code (particularly for other active learning algorithms) are adapted from their respective papers. 
