#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=xgpd[0,2-4],amdgpu[0-2]

# arguments: run_experiments_main.sh exp{n} {number of rounds to repeat}

SCRIPTPATH="~/al-ntk/experiments"

echo "script path = ${SCRIPTPATH}"

# get the variables from the shared experiment_cases.sh folder
source ${SCRIPTPATH}/experiment_cases.sh

alg="${alglist[$SLURM_ARRAY_TASK_ID]}"

testfolder=~/results_main/$1
train=${3:-true}
cpucount=${4:-1}

echo "args = $1 $2 $3 $4"

exp=${1}ens
echo "experiment_setup = ${!exp}"
echo "alg = ${alg}"

mkdir -p $testfolder

eval "$(conda shell.bash hook)"
conda activate ntk

ulimit -u 50000

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1  --xla_force_host_platform_device_count=${cpucount}"

TMPDIR=`mktemp -d`

for i in $(eval echo {1..$2}); do

  echo "--------- Loop $i ---------"

  # run script
  #root_dir=$1
  #exps=$2
  #al_alg=$3
  #train=${4:-true}
  #srun bash "${SCRIPTPATH}/run_experiments_2.sh" $TMPDIR $1 $alg true

  root_dir=$TMPDIR

  start_time=$(date +"%Y-%m-%d-%H-%M-%S")
  save_dir="${root_dir}"
  prefix="${start_time}"
  srun python "${SCRIPTPATH}/al_ensemble.py" results_dir="${save_dir}/${prefix}" ${!exp} al/alg="${alg}"

  randzip="${alg}-${start_time}.tgz"

  cd $TMPDIR

  tar -zcf ../$randzip .

  mv ../$randzip $testfolder

  cd ..

  rm -rf $TMPDIR/*

done

rm -rf $TMPDIR

conda deactivate