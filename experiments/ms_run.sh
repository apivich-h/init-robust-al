export CUDA_VISIBLE_DEVICES=$3
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

# for i in 1 2 3 4 5 6 7 8 9 10; do 
for i in 1 2 3 4 5; do 

    for alg in ms_ad nw_ad ev_ad; do
        
        python al-ntk/experiments/ms_run.py "$1" $2 20 ${alg}$4; 

    done

done