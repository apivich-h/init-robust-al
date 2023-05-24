#!/bin/bash

# arguments: run_experiments_main.sh exp{n} {number of rounds to repeat}

# CHANGE DEFAULT SCRIPT PATH HERE
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# CHANGE DEFAULT RESULTS PATH HERE
RESULTPATH=$( pwd "~/results_main" )/results_main

echo "script path = ${SCRIPTPATH}"

# get the variables from the shared experiment_cases.sh folder
source ${SCRIPTPATH}/experiment_cases.sh

train=${3:-true}
cpucount=${4:-1}
alg="${alglist[$5]}"

testfolder=${RESULTPATH}/$1

exp=${1}ens
echo "experiment_setup = ${!exp}"

# eval "$(conda shell.bash hook)"
# conda activate ntk

ulimit -u 50000

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_force_host_platform_device_count=${cpucount}"

for i in $(eval echo {1..$2}); do

  echo "--------- Loop $i ---------"

  start_time=$(date +"%Y-%m-%d-%H-%M-%S")
  prefix="${start_time}"
  echo "run ${SCRIPTPATH}/al_ensemble.py results_dir=${testfolder}/${prefix} ${!exp} al/alg=${alg}"
  python "${SCRIPTPATH}/al_ensemble.py" results_dir="${testfolder}/${prefix}" ${!exp} al/alg="${alg}"

done

# conda deactivate