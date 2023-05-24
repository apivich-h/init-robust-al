#!/bin/sh
#SBATCH --job-name=approx
#SBATCH --partition=long
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=2
# #SBATCH --mem=40G
#SBATCH --exclude=amdgpu[0-2]

# indexing method
# [initial set select method] [dataset] [dataset] [init size] [init size] [next batch sz] [next batch sz]

eval "$(conda shell.bash hook)"
conda activate ntk

ulimit -u 5000

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1  --xla_force_host_platform_device_count=2"

timemins=1060

# 0 = import
# 1 = set up everything
# 2 = select best batch
# 3 = train full
# 10 = run just to get criterion values
# 20 = run the multi training rounds
runstage=20

TMPDIR=`mktemp -d`

testfolder=~/results_regr

mkdir -p $testfolder

ff=$TMPDIR/TESTCASE_$SLURM_ARRAY_TASK_ID

echo "Temp folder = $ff"

echo "Running mode = $runstage"

mkdir -p $ff

if [ -f ~/results/TESTCASE_$SLURM_ARRAY_TASK_ID.tgz ]; then
  echo "copy files from ~/results/TESTCASE_$SLURM_ARRAY_TASK_ID to be used"

  cp ~/results/TESTCASE_$SLURM_ARRAY_TASK_ID.tgz $TMPDIR

  cd $TMPDIR

  tar -zxf TESTCASE_$SLURM_ARRAY_TASK_ID.tgz
fi

cd ~

srun python ~/al-ntk/experiments/regr_batch_run.py $ff $SLURM_ARRAY_TASK_ID $timemins $runstage

conda deactivate

cd $TMPDIR

tar -zcf TESTCASE_$SLURM_ARRAY_TASK_ID.tgz TESTCASE_$SLURM_ARRAY_TASK_ID

cp TESTCASE_$SLURM_ARRAY_TASK_ID.tgz $testfolder

cd ..

rm -rf $TMPDIR