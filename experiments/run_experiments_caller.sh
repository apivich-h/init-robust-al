#!/bin/bash

# command: ./run_experiments_caller.sh [experiment number] [experiments mode] [which gpu id to use] [also do training after AL (optional)]
# example: ./run_experiments_caller.sh exp20 51 7 {true}

SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Script path = ${SCRIPTPATH}"

exp=$1
mode=$2
cuda_visible=$3
custom_arr=$4
train=true

echo "args = $1 $2 $3 $4"

if [[ ! -z "$cuda_visible" ]] 
then

    export CUDA_VISIBLE_DEVICES=${cuda_visible}

    echo "using GPUs with ID ${cuda_visible}"

fi

cheap_arr="0,1"
medium_arr="3"  # or 2,3,4
expens_arr="7,8"  # or 5,7,8
ntkgp_arr="9,10,12,13,14,16"  # or 9,10,11,12,13
ntkgp_emp_arr="18,19,21,24,25,27"
ntkgp_ens_arr="29,30,31"


####################################
########## SLURM RUNNING ###########
####################################

if [[ $mode -eq 111 ]]  # SLURM jobs - small
then

    # medium AL experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=4 --array=${medium_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 4

    # cheap AL experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --array=${cheap_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 1

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 121 ]]  # SLURM jobs - large, but shorter time
then

    # expensive AL experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=16 --mem=250G --array=${expens_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 16

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 122 ]]  # SLURM jobs - large, but longer time
then

    # expensive AL experiments
    sbatch --partition=long --job-name=al-${exp} --time=1-12:00:00 --cpus-per-task=32 --mem=250G --array=${expens_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 131 ]]  # SLURM jobs - NTKGP, but shorter time
then

    # NTKGP experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=16 --mem=250G --array=${ntkgp_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 16

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 141 ]]  # SLURM jobs - NTKGP empirical, but shorter time
then

    # NTKGP experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=32 --mem=250G --array=${ntkgp_emp_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 142 ]]  # SLURM jobs - NTKGP empirical, but longer time
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=1-12:00:00 --cpus-per-task=32 --mem=250G --array=${ntkgp_emp_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 151 ]]  # SLURM jobs - NTKGP ensemble tests, but shorter time
then

    # NTKGP experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=32 --mem=250G --array=${ntkgp_ens_arr} ${SCRIPTPATH}/run_experiments_slurm_ens.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 152 ]]  # SLURM jobs - NTKGP ensemble tests, but longer time
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=1-12:00:00 --cpus-per-task=32 --mem=250G --array=${ntkgp_ens_arr} ${SCRIPTPATH}/run_experiments_slurm_ens.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 161 ]]  # SLURM jobs - custom array
then

    # NTKGP experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=8 --mem=160G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 162 ]]  # SLURM jobs - NTKGP empirical, but medium time
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=12:00:00 --cpus-per-task=8 --mem=160G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 163 ]]  # SLURM jobs - NTKGP empirical, but longer time
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=3-0:00:00 --cpus-per-task=16 --mem=160G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 171 ]]  # SLURM jobs - NTKGP ensemble tests, custom array
then

    # NTKGP experiments
    sbatch --partition=medium --job-name=al-${exp} --time=180 --cpus-per-task=4 --mem=40G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm_ens.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 172 ]]  # SLURM jobs - NTKGP ensemble tests, custom array
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=12:00:00 --cpus-per-task=8 --mem=160G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm_ens.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue

elif [[ $mode -eq 173 ]]  # SLURM jobs - NTKGP ensemble tests, custom array
then

    # NTKGP experiments
    sbatch --partition=long --job-name=al-${exp} --time=1-12:00:00 --cpus-per-task=32 --mem=250G --array=${custom_arr} ${SCRIPTPATH}/run_experiments_slurm_ens.sh $exp 5 $train 32

    sleep 1

    echo "===> squeue <==="

    squeue


######################################
########## REGULAR RUNNING ###########
######################################

elif [[ $mode -eq 211 ]]  # non-SLURM jobs - run only small algorithms
then

    algs2="${cheap_arr},${medium_arr}"
    # algs2="${cheap_arr},${medium_arr},${expens_arr}"

    IFS=','

    for c in ${algs2}; do

        # cheap + medium AL experiments
        ${SCRIPTPATH}/run_experiments_script.sh $exp 5 $train 8 $c
        
    done

elif [[ $mode -eq 221 ]]  # non-SLURM jobs - run only large algorithms
then

    algs2="${expens_arr}"

    IFS=','

    for c in ${algs2}; do

        ${SCRIPTPATH}/run_experiments_script.sh $exp 5 $train 32 $c
        
    done

elif [[ $mode -eq 231 ]]  # non-SLURM jobs - run only NTKGP (theoretical) algorithms
then

    algs2="${ntkgp_arr}"

    IFS=','

    for c in ${algs2}; do

        # NTKGP (non-empirical) AL experiments
        ${SCRIPTPATH}/run_experiments_script.sh $exp 5 $train 32 $c
        
    done

elif [[ $mode -eq 241 ]]  # non-SLURM jobs - run only NTKGP (empirical) algorithms
then

    algs1="${ntkgp_emp_arr}"

    IFS=','

    for c in ${algs1}; do

        # NTKGP experiments
        ${SCRIPTPATH}/run_experiments_script.sh $exp 5 $train 32 $c

    done

elif [[ $mode -eq 251 ]]  # non-SLURM jobs - run only NTKGP ensemble algorithms
then

    IFS=','

    for c in ${ntkgp_ens_arr}; do

        # NTKGP experiments
        ${SCRIPTPATH}/run_experiments_script_ens.sh $exp 5 $train 32 $c

    done

elif [[ $mode -eq 261 ]]  # non-SLURM jobs - run custom indices
then

    IFS=','

    for c in ${custom_arr}; do

        # NTKGP experiments
        ${SCRIPTPATH}/run_experiments_script.sh $exp 5 $train 32 $c

    done

elif [[ $mode -eq 271 ]]  # non-SLURM jobs - run NTKGP ensemble algorithms, custom array
then

    IFS=','

    for c in ${custom_arr}; do

        # NTKGP experiments
        ${SCRIPTPATH}/run_experiments_script_ens.sh $exp 5 $train 32 $c

    done

else

    echo "Invalid mode"

fi
