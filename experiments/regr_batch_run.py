import argparse
import os
import time
import signal
import tqdm

""" ARGPARSE """
parser = argparse.ArgumentParser()
parser.add_argument("file_prefix")
parser.add_argument("case_idx", type=int)
parser.add_argument("time_limit_mins", type=int)
parser.add_argument("stage", type=int)
args = parser.parse_args()

""" END ARGPARSE """

""" TIMER """


# For leaving program before timer runs out
def handler(signum, frame):
    exit()


signal.signal(signal.SIGALRM, handler)
signal.alarm(args.time_limit_mins * 60)

# 0 = import
# 1 = set up everything
# 2 = select best batch
# 3 = train full
# 10 = run just to get criterion values
# 20 = run the multi training rounds
# 30 = run multi training rounds with alternate loss (ce or bce)
# 40 = run all repeated training again
# 100+ = run with criterion again
run_stage = args.stage

""" END TIMER """

# ======================================================================================

t = time.time()

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
from jax.config import config

config.update("jax_enable_x64", True)

from al_ntk.dataset import MockFromModel, MockUnbalancedCluster, MockMismatched, UCI
from al_ntk.model import NNModel, MLPJax
from al_ntk.gp import FullRankGP, FIC
from al_ntk.utils.nn_training_jax import train_mlp
from al_ntk.al_algorithms import RandomMethod, NTKGPApproxMethod  #, BADGEMethod, GreedyCentreMethod
from al_ntk.al_algorithms.ntkgp_sampling import criterion_ev, criterion_mi, criterion_mv, \
    criterion_vol, criterion_ent, criterion_ent_masked, criterion_90v

print(f'JAX available resources = {jax.devices()}')
print(f'JAX using = {jax.default_backend()}')

t = time.time() - t
print(f'Library load time = {t}')

if run_stage < 1:
    signal.alarm(0)
    exit()

# ======================================================================================

file_prefix = args.file_prefix
case_idx = args.case_idx

case_idx_list = {
    0:  "10-1-000-002",

    # ===================

    1:  "10-1-000-020",
    2:  "10-1-020-020",
    3:  "11-1-000-020",
    4:  "11-1-020-020",

    5:  "12-1-000-020",
    6:  "12-1-020-020",
    7:  "13-1-000-020",
    8:  "13-1-020-020",
    
    # ===================

    9:  "40-1-000-020",  # boston
    10: "40-1-020-020",
    11: "42-1-000-020",  # wine
    12: "42-1-020-020",
    13: "43-1-000-020",  # energy
    14: "43-1-020-020",
    15: "44-1-000-020",  # kin8nm
    16: "44-1-020-020",
    17: "50-1-000-020",  # optdigits
    18: "50-1-020-020",
    19: "51-1-000-020",  # yeast
    20: "51-1-020-020",
    21: "52-1-000-020",  # image_segmentation
    22: "52-1-020-020",
    23: "56-1-000-020",  # image_segmentation_mismatched
    24: "56-1-020-020",
    25: "60-1-000-020",  # breast_cancer_wisconsin
    26: "60-1-020-020",
    27: "61-1-000-020",  # spambase
    28: "61-1-020-020",
    29: "65-1-000-020",  # breast_cancer_wisconsin_mismatched
    30: "65-1-020-020",
    31: "66-1-000-020",  # spambase_mismatched
    32: "66-1-020-020",
    33: "45-1-000-020",  # yacht
    34: "45-1-020-020",
    35: "46-1-000-020",  # concrete
    36: "46-1-020-020",
    37: "47-1-000-020",  # naval
    38: "47-1-020-020",
    39: "48-1-000-020",  # power
    40: "48-1-020-020",
    41: "49-1-000-020",  # protein
    42: "49-1-020-020",
    43: "39-1-000-020",  # optdigits_regr
    44: "39-1-020-020",
    
    61: "56-1-040-020",  # image_segmentation_mismatched (again)
    62: "56-1-060-020",  # image_segmentation_mismatched (again)
    63: "44-1-040-020",  # kin8nm (again)
    64: "51-1-040-020",  # yeast (again)
    65: "50-1-040-020",  # optdigits (again)
    66: "52-1-040-020",  # image_segmentation (again)
    67: "48-1-040-020",  # power (again)
    68: "60-1-010-020",  # breast_cancer_wisconsin
    69: "65-1-010-020",  # breast_cancer_wisconsin_mismatched
}

if case_idx > 99:
    use_empirical_kernel = True
    case_idx = case_idx - 100
else:
    use_empirical_kernel = False

case_idx_str = case_idx_list[case_idx].split('-')
dset = int(case_idx_str[0])
pts_init_selected_method = int(case_idx_str[1])
init_select_sz = int(case_idx_str[2])
next_batch_selection_sz = int(case_idx_str[3])

width = 512
activation = 'relu'

lr = 1e-2
sigma_noise = 1e-3

model_bias = 0.1

inducing_pt_method = 'kmeans'

regen_model_every_time = False
rand_train_rounds = 20
train_reps = 50


def model_gen_fn_from_data(data_t, hidden_count):
    def model_gen_fn():
        return MLPJax(in_dim=data_t.input_dimensions(), out_dim=data_t.output_dimensions(),
                      hidden_count=hidden_count, width=width, activation=activation,
                      use_empirical_kernel=use_empirical_kernel,
                      W_std=1., b_std=model_bias)

    return model_gen_fn


def model_gen_fn_from_dimensions(in_dim, out_dim, hidden_count):
    def model_gen_fn():
        return MLPJax(in_dim=(in_dim,), out_dim=out_dim,
                      hidden_count=hidden_count, width=width, activation=activation,
                      use_empirical_kernel=use_empirical_kernel,
                      W_std=1., b_std=model_bias)

    return model_gen_fn


if dset == 10:
    dset_name = 'mock_reg_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    data = MockFromModel(fn=model_gen(), n=1000, noise=1e-3, distribution='uniform_ball', problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 11:
    dset_name = 'mock_reg_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    data = MockFromModel(fn=model_gen(), n=1000, noise=2e-1, distribution='uniform_ball', problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 12:
    dset_name = 'mock_reg_unbal_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 1e-3 * r.randn(xs.shape[0], 1)
    data = MockMismatched(fn=fn, dim=8, n=1000, mean_dist=0.5, problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 13:
    dset_name = 'mock_reg_unbal_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 2e-1 * r.randn(xs.shape[0], 1)
    data = MockMismatched(fn=fn, dim=8, n=1000, mean_dist=0.5, problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 14:
    dset_name = 'mock_reg_clustered_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 1e-3 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=False, seed=51,
                                 problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 15:
    dset_name = 'mock_reg_clustered_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 2e-1 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=False, seed=51,
                                 problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 16:
    dset_name = 'mock_reg_clustered_mismatch_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 1e-3 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=True, seed=51,
                                 problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 17:
    dset_name = 'mock_reg_clustered_mismatch_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 2e-1 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=16, n=1000, n_clusters=2, evenly_distribute_tests=True, seed=51,
                                 problem_type='regression')
    inducing_pts = 80
    training_steps = 1000

elif dset == 20:
    dset_name = 'mock_cla_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=8, hidden_count=2)
    data = MockFromModel(model=model_gen(), n=1000, noise=1e-3, distribution='uniform_ball',
                         problem_type='classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 21:
    dset_name = 'mock_cla_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=8, hidden_count=2)
    data = MockFromModel(fn=model_gen(), n=500, noise=1e-1, distribution='uniform_ball',
                         problem_type='classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 30:
    dset_name = 'mock_bcl_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    data = MockFromModel(fn=model_gen(), n=1000, noise=1e-3, distribution='uniform_ball',
                         problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 31:
    dset_name = 'mock_bcl_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=8, out_dim=1, hidden_count=2)
    data = MockFromModel(fn=model_gen(), n=1000, noise=1e-1, distribution='uniform_ball',
                         problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 34:
    dset_name = 'mock_bcl_clustered_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=16, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 1e-3 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=False, seed=51,
                                 problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000
    sigma_noise = 1e-3

elif dset == 35:
    dset_name = 'mock_bcl_clustered_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=16, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 2e-1 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=False, seed=51,
                                 problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 36:
    dset_name = 'mock_bcl_clustered_mismatch_low_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=16, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 1e-3 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=True, seed=51,
                                 problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000

elif dset == 37:
    dset_name = 'mock_bcl_clustered_mismatch_high_noise'
    model_gen = model_gen_fn_from_dimensions(in_dim=16, out_dim=1, hidden_count=2)
    m = model_gen()
    r = np.random.RandomState(42)
    fn = lambda xs: np.array(m(xs)) + 2e-1 * r.randn(xs.shape[0], 1)
    data = MockUnbalancedCluster(fn=fn, dim=8, n=1000, n_clusters=2, evenly_distribute_tests=True, seed=51,
                                 problem_type='binary_classification')
    inducing_pts = 80
    training_steps = 1000
    
elif dset == 39:
    dset_name = 'optdigits_regr'
    data = UCI(dataset='optdigits_regr', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=0.05, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

elif dset == 40:
    dset_name = 'boston'
    data = UCI(dataset='boston', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 80
    training_steps = 2000

elif dset == 41:
    dset_name = 'yacht'
    data = UCI(dataset='yacht', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 80
    training_steps = 2000
    lr = 1e-1
    sigma_noise = 1e-3

elif dset == 42:
    dset_name = 'wine'
    data = UCI(dataset='wine', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 43:
    dset_name = 'energy'
    data = UCI(dataset='energy', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 44:
    dset_name = 'kin8nm'
    data = UCI(dataset='kin8nm', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=0.25, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 45:
    dset_name = 'yacht'
    data = UCI(dataset='yacht', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 46:
    dset_name = 'concrete'
    data = UCI(dataset='concrete', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 60
    training_steps = 2000

elif dset == 47:
    dset_name = 'naval'
    data = UCI(dataset='naval', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=0.2, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

elif dset == 48:
    dset_name = 'power'
    data = UCI(dataset='power', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=0.25, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 49:
    dset_name = 'protein'
    data = UCI(dataset='protein', normalise_x=True, normalise_y=True,
               whole_data_pool_prop=0.05, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 60
    training_steps = 2000

elif dset == 50:
    dset_name = 'optdigits'
    data = UCI(dataset='optdigits', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=0.4, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

elif dset == 51:
    dset_name = 'yeast'
    data = UCI(dataset='yeast', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 52:
    dset_name = 'image_segmentation'
    data = UCI(dataset='image_segmentation', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 55:
    dset_name = 'optdigits_mismatched'
    data = UCI(dataset='optdigits', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=0.5, test_split_prop=0.5,
               test_split_method='unbalanced_labels', test_split_param=0.35 + 0.03 * np.arange(10),
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

elif dset == 56:
    dset_name = 'image_segmentation_mismatched'
    data = UCI(dataset='image_segmentation', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=1., test_split_prop=0.5,
               test_split_method='unbalanced_labels', test_split_param=0.35 + 0.04 * np.arange(7),
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 60
    training_steps = 2000

elif dset == 60:
    dset_name = 'breast_cancer_wisconsin'
    data = UCI(dataset='breast_cancer_wisconsin', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=1., test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000

elif dset == 61:
    dset_name = 'spambase'
    data = UCI(dataset='spambase', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=0.5, test_split_prop=0.5,
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

elif dset == 65:
    dset_name = 'breast_cancer_wisconsin_mismatched'
    data = UCI(dataset='breast_cancer_wisconsin', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=1., test_split_prop=0.5,
               test_split_method='unbalanced_labels', test_split_param=np.array([0.6, 1.]),
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 30
    training_steps = 2000
    lr = 1e-1
    sigma_noise = 1e-2

elif dset == 66:
    dset_name = 'spambase_mismatched'
    data = UCI(dataset='spambase', normalise_x=True, normalise_y=False,
               whole_data_pool_prop=0.5, test_split_prop=0.5,
               test_split_method='unbalanced_labels', test_split_param=np.array([0.6, 1.]),
               random_state=42)
    model_gen = model_gen_fn_from_data(data_t=data, hidden_count=2)
    inducing_pts = 100
    training_steps = 2000

else:
    raise ValueError('Invalid dset parameter.')

# idiot check for model and data set correctly
assert model_gen()(data.xs_test).shape == (data.ys_test.shape[0], data.output_dimensions())  

# loss_fn = 'ce' if ((not data.treat_as_regression) and data.problem_type == 'classification') else 'mse'
loss_fn = 'mse'

if (os.path.isdir(f'{file_prefix}/data') and
        {'actual_xs_train.npy', 'actual_xs_test.npy',
         'actual_ys_train.npy', 'actual_xs_test.npy'}.issubset(set(os.listdir(f'{file_prefix}/data')))):
    data.reload_data(
        xs_train=np.load(f'{file_prefix}/data/actual_xs_train.npy'),
        ys_train=np.load(f'{file_prefix}/data/actual_ys_train.npy'),
        xs_test=np.load(f'{file_prefix}/data/actual_xs_test.npy'),
        ys_test=np.load(f'{file_prefix}/data/actual_ys_test.npy')
    )

else:
    os.makedirs(f'{file_prefix}/data', exist_ok=True)
    jnp.save(f'{file_prefix}/data/actual_xs_train.npy', data.xs_train)
    jnp.save(f'{file_prefix}/data/actual_ys_train.npy', data.ys_train)
    jnp.save(f'{file_prefix}/data/actual_xs_test.npy', data.xs_test)
    jnp.save(f'{file_prefix}/data/actual_ys_test.npy', data.ys_test)

# ======================================================================================

if not os.path.isfile(f'{file_prefix}/data/idxs_initial.npy'):

    if pts_init_selected_method == 1:
        pts_init_selected_method = 'random'
        rand = np.random.RandomState(42)
        perm = rand.permutation(data.train_count())
        idxs_initial = perm[:init_select_sz]
        idxs_unlabelled = perm[init_select_sz:]

    else:
        raise ValueError

    np.save(f'{file_prefix}/data/idxs_initial.npy', idxs_initial)
    np.save(f'{file_prefix}/data/idxs_unlabelled.npy', idxs_unlabelled)

else:
    idxs_initial = np.load(f'{file_prefix}/data/idxs_initial.npy')
    idxs_unlabelled = np.load(f'{file_prefix}/data/idxs_unlabelled.npy')

# ======================================================================================

record = f"""file_prefix = {file_prefix}
dset_name = {dset_name}
prob_type = {data.problem_type}
pts_init_selected_method = {pts_init_selected_method}
init_select_sz = {init_select_sz}
next_batch_selection_sz = {next_batch_selection_sz}
width = {width}
model_bias = {model_bias}
activation = {activation}
training_steps = {training_steps}
lr = {lr}
inducing_pts = {inducing_pts}
inducing_pt_method = {inducing_pt_method}
sigma_noise = {sigma_noise}
rand_train_rounds = {rand_train_rounds}
training_reps = {train_reps}
regen_model_every_time = {regen_model_every_time}
use_empirical_kernel = {use_empirical_kernel}
"""

print(record)
with open(f'{file_prefix}/record', 'w+') as f:
    f.write(record)


""" BEGIN TRAINING FULL DATA TO CHECK """

if not os.path.isfile(f'{file_prefix}/full_training/test_pred.npy'):
    print('Training whole training set for comparison')
    preds_all = []
    train_steps_recorded = []
    train_loss_all = []
    test_loss_all = []
    training_acc = []
    for r in range(1):
        m = model_gen()
        _, _, train_steps, train_loss, test_loss = train_mlp(model=m, dataset=data, epochs=training_steps,
                                                             learning_rate=lr, prog_bar=False, loss_fn=loss_fn)
        preds_all.append(m(data.xs_test))
        train_steps_recorded.append(train_steps)
        train_loss_all.append(train_loss)
        test_loss_all.append(test_loss)

    preds_all = jnp.array(preds_all)
    os.makedirs(f'{file_prefix}/full_training', exist_ok=True)
    jnp.save(f'{file_prefix}/full_training/train_steps.npy', np.array(train_steps_recorded))
    jnp.save(f'{file_prefix}/full_training/test_pred.npy', np.array(preds_all))
    jnp.save(f'{file_prefix}/full_training/train_loss.npy', np.array(train_loss_all))
    jnp.save(f'{file_prefix}/full_training/test_loss.npy', np.array(test_loss_all))

""" END TRAINING FULL DATA TO CHECK """

if run_stage < 2:
    signal.alarm(0)
    exit()

# ======================================================================================

model_for_kernel = model_gen()
empirical_ntk = model_for_kernel.get_ntk

al_criterions = {
    'ev':  ['frgp', 'sgp', 'sgpla'], 
    'mi':  ['frgp'], 
    'mv':  ['frgp', 'sgp'], 
    '90v': ['frgp', 'sgp'], 
    'vol': ['frgp', 'sgp', 'sgpla']
}

selected_batch_idxs_info = []

for criterion_str in al_criterions.keys():
    
    al_methods = al_criterions[criterion_str]
    
    for method_str in al_methods:

        case_prefix = f'{file_prefix}/batch_{criterion_str}-{method_str}'

        if (run_stage == 10) or (run_stage == 20):

            os.makedirs(case_prefix, exist_ok=True)

            t = time.time()

            lazy_eval = (criterion_str in {'ev', 'mi', 'vol'})
            
            alg = NTKGPApproxMethod(
                dataset=data, model=model_for_kernel, lazy_eval=False,
                criterion=criterion_str, posterior_approx_method=method_str,
                inducing_pt_method=inducing_pt_method, inducing_count=inducing_pts,
                sigma_noise=sigma_noise,
                use_train_set_for_test=True, test_prop_seen_max=(1000 if (criterion_str != 'mi') else 200), 
                test_prop_method='kmpp'
            )
            alg.preselect(idxs_initial)
            alg.get(next_batch_selection_sz)
            selected_batch_idxs = np.array(alg.last_batch_idxs)

            t = time.time() - t
            print(f'Best point {method_str} {criterion_str} selection time = {t}')
            with open(f'{case_prefix}/time', 'w+') as f:
                f.write(f'Best point {method_str} {criterion_str} selection time = {t}\n')

            np.save(f'{case_prefix}/batch_idxs.npy', selected_batch_idxs)
            np.save(f'{case_prefix}/selected_criterion.npy', np.array(alg.criterion_vals))

            open(f'{case_prefix}/done_flag_compute_idx', 'w+').close()

        else:

            print(f'Point selection already done')
            selected_batch_idxs = np.load(f'{case_prefix}/batch_idxs.npy')

        selected_batch_idxs_info.append((case_prefix, selected_batch_idxs))

if run_stage < 3:
    signal.alarm(0)
    exit()

# ======================================================================================

"""  BEGIN EXECUTION  """

Xn = jnp.array(data.xs_train)
Xt = jnp.array(data.xs_test)
yn, yt = data.generate_label_for_regression()
yn = jnp.array(yn)
yt = jnp.array(yt)

frgp = FullRankGP(kernel_fn=empirical_ntk, Xn=Xn, yn=yn, Xt=Xt, sigma_noise=sigma_noise,
                  use_train_set_for_test=False, keep_training_kernel_vals=True, only_track_diag=False)
sgp = FIC(kernel_fn=empirical_ntk, Xn=Xn, yn=yn, Xt=Xt, sigma_noise=sigma_noise,
          inducing_pts=inducing_pt_method, m=inducing_pts,
          use_train_set_for_test=False, keep_training_kernel_vals=True, only_track_diag=False)

jnp.save(f'{file_prefix}/sgp_inducing_inputs.npy', jnp.array(sgp.Xu))

""" PERFORM ALL OTHER APPROXIMATIONS BUT WITH MASKED_VALUES """

# sel = jnp.zeros(shape=data.train_count(), dtype=bool).at[idxs_initial].set(True)

# for (gp, gp_name) in [(frgp, 'frgp'), (sgp, 'sgp')]:

#     gp.update_train(indexs=idxs_initial)

#     train_mean = gp.get_train_posterior_mean()
#     test_mean, _ = gp.get_test_posterior()
#     jnp.save(f'{file_prefix}/{gp_name}_pseudolabel_train_modelout.npy', jnp.array(train_mean))
#     jnp.save(f'{file_prefix}/{gp_name}_pseudolabel_test_modelout.npy', jnp.array(test_mean))

#     if data.problem_type == 'classification':
#         embedding = jnp.eye(data.output_dimensions())
#         train_mean = embedding[jnp.argmax(train_mean, axis=1), :]
#         test_mean = embedding[jnp.argmax(test_mean, axis=1), :]
#     elif data.problem_type == 'binary_classification':
#         train_mean = jnp.sign(train_mean)
#         test_mean = jnp.sign(test_mean)

#     pseudolabels = train_mean.at[sel].set(yn[sel, :])
#     gp.update_labels(new_yn=pseudolabels, new_yt=test_mean)

#     jnp.save(f'{file_prefix}/{gp_name}_pseudolabel_train.npy', jnp.array(train_mean))
#     jnp.save(f'{file_prefix}/{gp_name}_pseudolabel_test.npy', jnp.array(test_mean))

for rnd in range(-len(selected_batch_idxs_info), rand_train_rounds):

    if (rnd >= 0) and (run_stage >= 100):
        break

    if rnd < 0:
        subfile_prefix, batch_idx = selected_batch_idxs_info[rnd]
        os.makedirs(subfile_prefix, exist_ok=True)
    else:
        subfile_prefix = f'{file_prefix}/batch_random_rnd{rnd}'
        os.makedirs(subfile_prefix, exist_ok=True)
        if os.path.isfile(f'{subfile_prefix}/batch_idxs.npy'):
            batch_idx = np.load(f'{subfile_prefix}/batch_idxs.npy')
        else:
            batch_idx = np.random.permutation(idxs_unlabelled)[:next_batch_selection_sz]
            np.save(f'{subfile_prefix}/batch_idxs.npy', batch_idx)

    print(f'For : {subfile_prefix}')

    """ BEGIN GP """

    if run_stage >= 10:

        for (gp, gp_name) in [(frgp, 'frgp'), (sgp, 'sgp')]:

            t = time.time()
            gp.update_train(indexs=idxs_initial)
            t = time.time() - t

            print(f'{gp_name} precompute time = {t}')
            with open(f'{subfile_prefix}/time', 'a+') as f:
                f.write(f'{gp_name} precompute time = {t}\n')

            print(f'Checking {gp_name} tests')
            for i in tqdm.trange(next_batch_selection_sz + 1):
                # x = batch_idx[i]
                # gp.incrementally_update_train(additional_indexs=jnp.arange(x, x + 1))
                gp.update_train(jnp.concatenate([idxs_initial, batch_idx[:i]]))

                tr_mask = jnp.ones(data.train_count()).at[gp.indexs].set(0.)

                gp_train_prior = gp.K_nn
                gp_mean_tr = gp.get_train_posterior_mean()
                gp_cov_tr = gp.get_train_posterior_covariance()
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_mean_tr.npy', gp_mean_tr)
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_cov_tr.npy', jnp.diag(gp_cov_tr))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_ev_tr.npy',
                         criterion_ev(ys=yn, mean=gp_mean_tr, var=jnp.diag(gp_cov_tr)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_mv_tr.npy',
                         criterion_mv(ys=yn, mean=gp_mean_tr, var=jnp.diag(gp_cov_tr)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_90v_tr.npy',
                         criterion_90v(ys=yn, mean=gp_mean_tr, var=jnp.diag(gp_cov_tr)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_mi_tr.npy',
                         criterion_mi(ys=yn, mean=gp_mean_tr, 
                                      cov=gp_cov_tr + sigma_noise**2 * jnp.eye(gp_cov_tr.shape[0]), 
                                      mask=tr_mask, prior=gp_train_prior))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_vol_tr.npy',
                         criterion_vol(ys=yn, mean=gp_mean_tr, var=jnp.diag(gp_cov_tr)))

                t = time.time()
                te_mask = jnp.ones(shape=data.test_count())
                gp_mean, gp_cov = gp.get_test_posterior()
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_mean_te.npy', gp_mean)
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_cov_te.npy', jnp.diag(gp_cov))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_ev_te.npy',
                         criterion_ev(ys=data.ys_test, mean=gp_mean, var=jnp.diag(gp_cov)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_mv_te.npy',
                         criterion_mv(ys=data.ys_test, mean=gp_mean, var=jnp.diag(gp_cov)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_90v_te.npy',
                         criterion_90v(ys=data.ys_test, mean=gp_mean, var=jnp.diag(gp_cov)))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_mi_te.npy',
                         criterion_ent(ys=data.ys_test, mean=gp_mean, 
                                       cov=gp_cov + (sigma_noise**2 * jnp.eye(gp_cov.shape[0]))))
                jnp.save(f'{subfile_prefix}/r{i}_{gp_name}_criterion_vol_te.npy',
                         criterion_vol(ys=data.ys_test, mean=gp_mean, var=jnp.diag(gp_cov)))
                t = time.time() - t
                with open(f'{subfile_prefix}/time', 'a+') as f:
                    f.write(f'{gp_name} time round {i} = {t}\n')

            """ END GP """

        open(f'{subfile_prefix}/done_flag_critvals', 'w+').close()

    # ======================================================================================

    if (run_stage >= 20) and not os.path.isfile(f'{subfile_prefix}/done_flag_trainingproc'):

        """ BEGIN FULL TRAINING """

        t = time.time()
        print('Training actual data')

        for i in tqdm.trange(next_batch_selection_sz + 1):

            selected = np.zeros(shape=data.train_count(), dtype=bool)
            selected[np.array(idxs_initial)] = True
            selected[np.array(batch_idx[:i])] = True
            selected = jnp.array(selected)

            train_preds = []
            train_losses = []
            test_preds = []
            test_losses = []
            for r in range(train_reps):
                m = model_gen()
                _, _, recorded_steps, train_loss, test_loss = train_mlp(model=m, dataset=data, training_subset=selected,
                                                                        epochs=training_steps, learning_rate=lr,
                                                                        prog_bar=False, loss_fn=loss_fn)
                train_preds.append(m(data.xs_train))
                train_losses.append(train_loss)
                test_preds.append(m(data.xs_test))
                test_losses.append(test_loss)

            jnp.save(f'{subfile_prefix}/r{i}_train_preds.npy', jnp.array(train_preds))
            jnp.save(f'{subfile_prefix}/r{i}_train_mse_loss.npy', jnp.array(train_losses))
            jnp.save(f'{subfile_prefix}/r{i}_test_preds.npy', jnp.array(test_preds))
            jnp.save(f'{subfile_prefix}/r{i}_test_mse_loss.npy', jnp.array(test_losses))

        t = time.time() - t
        print(f'Overall training time = {t}')
        with open(f'{subfile_prefix}/time', 'a+') as f:
            f.write(f'Overall training time = {t}\n')

        """ END FULL TRAINING """

        open(f'{subfile_prefix}/done_flag_trainingproc', 'w+').close()

# ======================================================================================

# if finish before time limit, cancel alarm
signal.alarm(0)
exit()

""" END EXECUTION """