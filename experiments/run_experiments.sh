#!/bin/bash

# USAGE: bash run_experiments.sh exp1,exp2,exp5,exp10 true 

exps=$1
train=${2:-true}

start_time=$(date +"%d-%m-%Y/%H-%M-%S")
save_dir="results/${start_time}"

exp1="al=v1 data=test model=mlp train=v1 use_cuda=false"

IFS=',' read -ra exps <<< "${exps}"
for exp in "${exps[@]}"; do
  python3 -m experiments.al results_dir="${save_dir}/${exp}" ${!exp}
   if [ "${train}" = "true" ]; then
     dirs=$(find ${save_dir} -type d -iname "${exp}*" )
     if [[ ${#dirs[@]} -eq 1 ]]; then
       python3 -m experiments.train --config-path "../${dirs[0]}/active_learning"
     else
       echo "num dirs w suffix ${save_dir}/${exp} neq 1"
     fi
   fi
done