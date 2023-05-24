# -------------------------------------------------------
# LIST OF ALGORITHMS

alglist=(\
    'random'                     # 0  \
    'kmeans-pp'                  # 1  \
    'active-coreset'                  \
    'badge'                      # 3  \
    'badge-randinit'                  \
    'batchbald'                  # 5  \
    'mlmoc-batched-randinit-l'   # 6  \
    'mlmoc-batched-randinit'     # 7  \
    'mlmoc-batched'                   \
    'ntkgp-frgp-ev'              # 9  \
    'ntkgp-frgp-mv'                   \
    'ntkgp-frgp-90v'                  \
    'ntkgp-frgp-mi'                   \
    'ntkgp-sgp-ev'                    \
    'ntkgp-sgp-mv'                    \
    'ntkgp-sgp-90v'                   \
    'ntkgp-sgpla-ev'                  \
    'ntkgp-sgpla-mv'                  \
    'ntkgp-frgp-ev-emp'          # 18 \
    'ntkgp-frgp-mv-emp'               \
    'ntkgp-frgp-90v-emp'              \
    'ntkgp-frgp-mi-emp'               \
    'ntkgp-frgp-ev-emp-train'         \
    'ntkgp-frgp-ev-emp-small'    # 23 \
    'ntkgp-sgp-ev-emp'                \
    'ntkgp-sgp-mv-emp'                \
    'ntkgp-sgp-90v-emp'               \
    'ntkgp-sgpla-ev-emp'         # 27 \
    'ntkgp-sgpla-mv-emp'         # 28 \
    'ntkgp-ensemble-ev'          # 29 \
    'mlmoc-n-seq'                # 30 \
    'mlmoc-n-batched'            # 31 \
    'mlmoc-n-batched-randinit'   # 32 \
)

# -------------------------------------------------------
# LIST OF EXPERIMENTS

# test dataset
exp0="data=uci-small data.params.dataset=test model=mlp-2hl train=uci"

# uci datasets
exp1="data=uci-small data.params.dataset=boston model=mlp-2hl train=uci"
exp3="data=uci-large data.params.dataset=kin8nm model=mlp-2hl train=uci"
exp4="data=uci-large data.params.dataset=optdigits model=mlp-2hl train=uci"
exp7="data=uci-large data.params.dataset=protein model=mlp-2hl train=uci"
exp8="data=uci-large data.params.dataset=concrete model=mlp-2hl train=uci"
exp9="data=uci-large data.params.dataset=power model=mlp-2hl train=uci"
exp10="data=uci-small data.params.dataset=wine model=mlp-2hl train=uci"
exp11="data=uci-large data.params.dataset=optdigits_regr model=mlp-2hl train=uci"
exp12="data=uci-large data.params.dataset=naval model=mlp-2hl train=uci"
exp13="data=uci-large data.params.dataset=test2 model=mlp-2hl train=uci"
exp14="data=uci-large data.params.dataset=test_sin data.params.normalise_y=false model=mlp-2hl train=uci"

# mnist - but with MLP
exp25="data=mnist-small-flatten model=mlp-2hl data.batch_sz=100 train=mnist"
exp26="data=mnist-small-flatten model=mlp-2hl data.batch_sz=200 train=mnist"
exp27="data=mnist-small-flatten model=mlp-2hl data.batch_sz=400 train=mnist"

# emnist
exp35="data=emnist model=mnist data.batch_sz=100 train=conv"
exp36="data=emnist model=mnist data.batch_sz=200 train=conv"
exp37="data=emnist model=mnist data.batch_sz=300 train=conv"
exp38="data=emnist model=mnist data.batch_sz=600 train=conv"

exp135="data=emnist-flatten model=mlp-3hl data.budget=800 data.batch_sz=100 train=mnist"
exp136="data=emnist-flatten model=mlp-3hl data.budget=800 data.batch_sz=200 train=mnist"
exp137="data=emnist-flatten model=mlp-3hl data.budget=800 data.batch_sz=400 train=mnist"
exp138="data=emnist-flatten model=mlp-3hl data.budget=800 data.batch_sz=800 train=mnist"

# emnist unbalanced
exp40="data=emnist-unbalanced model=emnist data.batch_sz=100 train=conv"

# svhn
exp85="data=svhn model=resnet-d1 data.batch_sz=100 train=conv al.alg.use_torch=true"
exp86="data=svhn model=resnet-d1 data.batch_sz=200 train=conv al.alg.use_torch=true"
exp87="data=svhn model=resnet-d1 data.batch_sz=400 train=conv al.alg.use_torch=true"
exp88="data=svhn model=resnet-d1 data.batch_sz=800 train=conv al.alg.use_torch=true"

exp185="data=svhn model=resnet-d1 data.budget=2000 data.batch_sz=500 train=conv al.alg.use_torch=true"

exp386="data=svhn model=resnet18 data.batch_sz=200 data.budget=1000 train=res al.alg.use_torch=true"
exp387="data=svhn model=resnet18 data.batch_sz=1000 data.budget=5000 train=res al.alg.use_torch=true"

# cifar100
exp91="data=cifar100 model=resnet-d2 data.budget=1000 data.batch_sz=200 train=conv al.alg.use_torch=true"
exp95="data=cifar100 model=resnet-d2 data.budget=2000 data.batch_sz=500 train=conv al.alg.use_torch=true"

# -------------------------------------------------------