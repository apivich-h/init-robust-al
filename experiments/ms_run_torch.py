from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

import os
from datetime import datetime

import torch
torch.set_default_dtype(torch.float64)

print(torch.cuda.is_available())

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import lognorm, multivariate_normal
from scipy.linalg import block_diag
import tqdm
import neural_tangents as nt
from datetime import datetime
import pickle
import sys
import random
import string

import torch.nn as nn
import torch.optim as optim

from al_ntk.dataset import MockUnbalancedCluster, MockFromModel, UCI, MockMismatched, EMNIST, SVHN
from al_ntk.model.ensemble import EnsembleModelTorch, EnsembleModelJax
from al_ntk.al_algorithms import NTKGPApproxMethod, NTKGPEnsembleMethod
from al_ntk.utils.nn_training_torch import train_torch
from al_ntk.utils.nn_training_jax import train_mlp
from al_ntk.utils.torch_dataloader import get_torch_dataset, data_loaders
from al_ntk.utils.maps import optim_map
from al_ntk.gp.frgp import FullRankGP

root_folder = sys.argv[1]
dset = sys.argv[2]
batch_sz = int(sys.argv[3])

do_ms = ('ms' in sys.argv[4])
do_naswot = ('nw' in sys.argv[4])
use_ntk = ('ntk' in sys.argv[4])
small_fold = ('sf' in sys.argv[4])
use_adam = ('ad' in sys.argv[4])
use_momentum = ('mo' in sys.argv[4])
use_rbf = ('rbf' in sys.argv[4])
train_all = ('ta' in sys.argv[4])

score_tradeoff = 1.0

# printing lowercase
letters = string.ascii_lowercase
rand_str = ''.join(random.choice(letters) for i in range(10))

date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
folder = f'{root_folder}/{dset}-bsz{batch_sz:03d}-{sys.argv[4]}-{score_tradeoff}-{date_time}_{rand_str}'
os.makedirs(folder, exist_ok=True)

print('folder =', folder)
print('dset =', dset)
print('batch_sz =', batch_sz)
print('do_ms =', do_ms)
print('do_naswot =', do_naswot)
print('use_ntk =', use_ntk)
print('small_fold =', small_fold)
print('use_rbf =', use_rbf)
print('train_all =', train_all)
print('score_tradeoff =', score_tradeoff)

os.makedirs(folder, exist_ok=True)

rounds = 10

random_first_batch_count = batch_sz

train_dict = {
    'epochs': 1000,
    'data_loader_args': {
        'batch_size': 512,
        'test_batch_size': 512,
        'shuffle': True,
        'in_class_val': 1.,
        'not_in_class_val': -1.,
    } 
}
    
lr = 0.01
    
print('lr =', lr) 

activations = ['relu', 'gelu', ('leaky_relu', 0.1), 'erf']
if use_rbf:
    activations.append(('rbf', 0.1))


whole_data_pool_prop = 0.3 if (dset == 'protein') else 1.
data = UCI(dataset=dset, whole_data_pool_prop=whole_data_pool_prop, test_split_prop=0.5)
print('data dimensions', data.xs_train.shape, data.ys_train.shape, data.xs_test.shape, data.ys_test.shape)
os.makedirs(folder, exist_ok=True)

if do_ms or do_naswot:
    model_params = {
        'in_dim': data.input_dimensions(), 
        'out_dim': 1, 
        'family': 'mlp', 
        'model_init_weights': 1, 
        'use_empirical_kernel': do_naswot,
        'min_weight_to_use': 1e-6, 
        'kernel_batch_sz': 256,
        'dropout_rate': (0.1,),
        'mlp_activations': activations,
        'mlp_hidden_layers': (1, 2, 3, 4,), 
        'mlp_width': (512,), 
        'mlp_bias': (0.1,),
        'parametrisation': 'ntk',
    }
    start_with_gp = True
    
else:
    model_params = {
        'in_dim': data.input_dimensions(), 
        'out_dim': 1, 
        'family': 'mlp', 
        'model_init_weights': 0, 
        'use_empirical_kernel': False,
        'min_weight_to_use': 1e-6, 
        'kernel_batch_sz': 256,
        'dropout_rate': (0.1,),
        'mlp_activations':('relu',), 
        'mlp_hidden_layers': (2,), 
        'mlp_width': (512,), 
        'mlp_bias': (0.1,),
        'parametrisation': 'ntk',
    }
    start_with_gp = True
    
sigma_noise = 1e-2

ensemble = EnsembleModelTorch.construct_ensemble(**model_params)
ensemble.reweight_models()


if do_naswot:
    ms_criterion = 'naswot'
    criterion_params = {
        'lr': lr
    }

else:
    ms_criterion = 'el'
    criterion_params = {
        'score_tradeoff': score_tradeoff,
        'fold_prop': 0.1 if small_fold else 0.3,
        'folds': 100,
        'use_nn_variance': not use_ntk,
    }

alg = NTKGPEnsembleMethod(
    dataset=data, model=ensemble, ms_criterion=ms_criterion, criterion_params=criterion_params, start_with_gp=start_with_gp,
    criterion='ev', lazy_eval=True, sigma_noise=sigma_noise, posterior_approx_method='frgp',
    use_train_set_for_test=True, test_prop_seen_max=10000, test_prop_method='kmpp', check_per_round=1000,
    keep_training_kernel_vals=True, reinit_gp=False,
)


curr_sz = 0

for r in range(rounds):
    
    print(f'######## ROUND {r + 1} of {rounds} ########')
    
    # if (r == 0) and (batch_sz < 100):
    #     # first batch slightly bigger than otherwise
    #     alg.get(2 * batch_sz)
    #     curr_sz += 2 * batch_sz
    # else:
    alg.get(batch_sz)
    curr_sz += batch_sz    
    
    sz_folder = f'{folder}/size={curr_sz}'
    os.makedirs(sz_folder, exist_ok=True)
    with open(f'{sz_folder}/model_scoring', 'wb+') as f:
        pickle.dump(alg.last_iter_data, f)
    
    best_model = alg.last_iter_data['best_model']
    
    if train_all:
        idxs = range(len(ensemble.models))
    else:
        idxs = [best_model]
    
    print('Repeated training stage')
    for i in idxs:

        m = ensemble.models[i]
        
        outputs = []
        losses = []
        
        # do 100 in case of training error
        for rnd in range(100):
            
            train_epochs = train_dict['epochs']
            train_batch_sz = train_dict['data_loader_args']['batch_size']
                
            m.init_weights()
            optz = optim.Adam(m.parameters(), lr=lr)
            
            train_torch(
                model=m,
                train_loader=data_loaders(
                    dataset=data,
                    selected=alg.get_selected(),
                    batch_size=256,
                    device='cuda'
                )[0],
                optimizer=optz,
                criterion='mse',
                progbar=False,
                epochs=train_epochs,
            )
            output = m.call_np(data.xs_test)
                
            loss = (output - data.ys_test) ** 2
            if np.isnan(output).any():
                continue
        
            outputs.append(output)
            losses.append(loss)
            
            # only collect 50 samples
            if len(losses) >= 50:
                break
            
        print(f'Model {i} - mean_loss = {np.mean(np.array(losses))}')    
        subfolder = f'{sz_folder}/model{i}'
        os.makedirs(subfolder, exist_ok=True)
        np.save(f'{subfolder}/test_preds', np.array(outputs))
        np.save(f'{subfolder}/test_losses', np.array(losses))
        
    print('-----------------')