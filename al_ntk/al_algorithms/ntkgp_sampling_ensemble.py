from typing import Callable, Union, Dict
from functools import partial

import numpy as np
import jax
from jax import random
import torch
import jax.numpy as jnp
import tqdm
import time
from sklearn.cluster import kmeans_plusplus
from scipy.stats import lognorm, multivariate_normal
from scipy.linalg import block_diag

from al_ntk.dataset import Dataset
from al_ntk.model.ensemble import EnsembleModelTorch, EnsembleModelJax
from .base_al_algorithm import ALAlgorithm
from .ntkgp_sampling import NTKGPApproxMethod
from .kmeans_pp import KMeansPlusPlusMethod
from al_ntk.gp import FullRankGP, FIC
from al_ntk.utils.entropy_helper import max_entropy_selector, max_entropy_selector_from_fn
from al_ntk.utils.kernels_helper import compute_kernel_in_batches, approximate_full_kernel_as_block_diag
from al_ntk.utils.maps import optim_map
from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.nn_training_torch import train_torch


def leave_one_out(ys, cov, sigma):
    cov_inv = np.linalg.inv(np.asarray(cov) + sigma**2 * np.eye(cov.shape[0]))
    cov_inv_diag = np.diag(cov_inv)
    cov_inv_ys = cov_inv @ ys
    llhs = []
    n = ys.shape[1]
    for i in range(ys.shape[0]):
        yi_minus_mui = cov_inv_ys[i] / cov_inv_diag[i]
        vari = max(1 / cov_inv_diag[i], sigma**2)
        logp = -0.5 * np.log(vari) - np.sum(0.5 * yi_minus_mui**2 / vari)
        llhs.append(logp)
    return np.mean(llhs)


def compute_naswot_sim(net, inputs, split_data=1, loss_fn=None):
    device = inputs.device
    net.zero_grad()
    
    net.K = np.zeros((inputs.shape[0], inputs.shape[0]))
    def counting_forward_hook(module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = inp.float()
        K = x @ x.t()
        net.K += K.detach().cpu().numpy()

    hooks = []
    for module in list(net.modules())[::-1]:
        name = str(type(module))
        if ('ReLU' in name) or ('GELU' in name) or ('LambdaLayer' in name):  # @TODO: better check method
            h = module.register_forward_hook(counting_forward_hook)
            hooks.append(h)
            break

    _ = net.forward(inputs)
    s, ld = np.linalg.slogdet(net.K)
    for h in hooks:
        h.remove()
    return ld


class NTKGPEnsembleMethod(ALAlgorithm):
    
    def __init__(self, dataset: Dataset, model: Union[EnsembleModelTorch, EnsembleModelJax], 
                 ms_criterion: str = 'llh', random_first_batch_count: int = None, start_with_gp: bool = False,
                 criterion: Union[str, Callable] = 'ev', criterion_params: Dict = None, lazy_eval: bool = True,
                 posterior_approx_method: str = 'frgp', sigma_noise: float = 1e-3, reinit_gp: bool = False,
                 inducing_count: int = 100, inducing_pt_method: str = 'entropy', not_in_class_val: float = -1.,
                 use_train_set_for_test: bool = False, test_prop_seen_max: int = 5000, 
                 test_prop_method: str = 'random', keep_training_kernel_vals: bool = False, 
                 check_per_round: int = 6000, approx_kernel_as_block_diag: int = -1,
                 train: dict = None, optim: dict = None, use_cuda: bool = False):
        super().__init__(dataset, model)
                
        self.sigma_noise = sigma_noise
        self.ms_criterion = ms_criterion
        self.start_with_gp = start_with_gp
        self.criterion_params = criterion_params

        if train is not None or optim is not None:
            self.do_training = True
            self.train_epochs = train['epochs']
            self.train_batch_sz = train['data_loader_args']['batch_size']
            self.optimiser_generator = lambda params: optim_map[optim['name']](params, **optim['params'])
            self.in_class_val = train['data_loader_args']['in_class_val']
            self.not_in_class_val = train['data_loader_args']['not_in_class_val']
        else:
            self.do_training = False
        
        self.ntkgp_al_gen = lambda i: NTKGPApproxMethod(
            dataset=self.dataset,
            model=self.model[i],
            criterion=criterion,
            lazy_eval=lazy_eval,
            posterior_approx_method=posterior_approx_method,
            sigma_noise=sigma_noise,
            reinit_gp=reinit_gp,
            inducing_count=inducing_count,
            inducing_pt_method=inducing_pt_method,
            not_in_class_val=not_in_class_val,
            use_train_set_for_test=use_train_set_for_test,
            test_prop_seen_max=test_prop_seen_max,
            test_prop_method=test_prop_method,
            keep_training_kernel_vals=keep_training_kernel_vals,
            check_per_round=check_per_round,
            approx_kernel_as_block_diag=approx_kernel_as_block_diag,
            random_first_batch_count=random_first_batch_count,
            train=None,
            optim=None,
            use_cuda=use_cuda
        )
        
    def preselect(self, idxs: np.ndarray):
        self.selected[idxs] = True
        
    def get(self, n: int):
        
        selected_prev = jnp.copy(self.selected)
        
        if self.do_training and self.selected.any():
            self._train_model()
        else:
            self.model.init_weights()
        
        if self.start_with_gp or self.selected.any():
            best_idx = np.argmax(self.model.get_weights())
            alg = self.ntkgp_al_gen(best_idx)
            if self.selected.any():
                alg.preselect(np.argwhere(self.selected).flatten())
        else:
            alg = KMeansPlusPlusMethod(dataset=self.dataset, model=None)

        alg.get(n)
        
        self.selected = alg.selected
        selected_round = self.selected & (~selected_prev)
            
        if self.ms_criterion.startswith('el'):
            
            score_tradeoff = self.criterion_params.get('score_tradeoff', 1.)
            fold_prop = self.criterion_params.get('fold_prop', 0.1)
            folds = self.criterion_params.get('folds', 50)
            use_nn_variance = self.criterion_params.get('use_nn_variance', False)
            verbose = self.criterion_params.get('verbose', False)
            
            xs = jnp.array(self.dataset.xs_train[alg.selected])
            ys = jnp.array(self.dataset.generate_label_for_regression()[0])[alg.selected]
            test_count = max(1, int(fold_prop * xs.shape[0]))
            if test_count == 1:
                folds = xs.shape[0]
                
            print(f'Using el criterion for model selection: score_tradeoff={score_tradeoff}, fold_test_count={test_count}, ' +
                  f'folds={folds}, use_nn_variance={use_nn_variance}')
            
            assert self.dataset.output_dimensions() == 1  # implement 1d output case only first
            
            scores_arr = [[] for _ in self.model]
            mean_var_arr = [[] for _ in self.model]
            
            Ks = [m.get_ntk(x1=xs) for m in self.model]
            if use_nn_variance:
                Ss = [m.get_nngp(x1=xs) for m in self.model]
            
            for f in (range(folds) if verbose else tqdm.trange(folds)):
                
                if verbose:
                    print(f'Fold {f+1}')
                
                if test_count > 1:
                    idxs = jnp.array(np.random.permutation(xs.shape[0]))
                    prev = idxs[test_count:]
                    curr = idxs[:test_count]
                else:
                    prev = jnp.array([i for i in range(xs.shape[0]) if i != f])
                    curr = jnp.array([f])
                
                ys_prev = ys[prev, :]
                ys_curr = ys[curr, :]
                
                # xs_prev = xs[idxs[test_count:]]
                # xs_curr = xs[idxs[:test_count]]
                # ys_prev = ys[idxs[test_count:]]
                # ys_curr = ys[idxs[:test_count]]
            
                for i, m in enumerate(self.model):
                    K_nn = Ks[i][prev, :][:, prev]
                    K_nt = Ks[i][prev, :][:, curr]
                    K_tt = Ks[i][curr, :][:, curr]
                    K_nn_inv = jnp.linalg.inv(K_nn + self.sigma_noise**2 * jnp.eye(K_nn.shape[0]))
                    mu = K_nt.T @ K_nn_inv @ ys_prev
                    
                    if use_nn_variance:
                        # S_nn = m.get_nngp(x1=xs_prev)
                        # S_nt = m.get_nngp(x1=xs_prev, x2=xs_curr)
                        # S_tt = m.get_nngp(x1=xs_curr)
                        S_nn = Ss[i][prev, :][:, prev]
                        S_nt = Ss[i][prev, :][:, curr]
                        S_tt = Ss[i][curr, :][:, curr]
                        A = S_nt.T @ K_nn_inv @ K_nt
                        Sigma = S_tt - A - A.T + (K_nt.T @ K_nn_inv @ S_nn @ K_nn_inv @ K_nt)
                    else:
                        Sigma = K_tt - K_nt.T @ K_nn_inv @ K_nt + self.sigma_noise**2 * jnp.eye(K_tt.shape[0])
                    
                    mean = jnp.mean((ys_curr - mu)**2)
                    var = jnp.mean(jnp.diag(Sigma))
                    score = - (mean + score_tradeoff * var)
                    if verbose:
                        print(f'Model {i:2d} | score = {score:.5f} | mean = {mean:.5f}, var = {var:.5f}')
                    
                    scores_arr[i].append(score)
                    mean_var_arr[i].append((mean, var))
                
            scores = [np.mean(s) for s in scores_arr]
            self.last_iter_data = {
                'scores_raw': scores_arr,
                'mean_var_arr': mean_var_arr,
            }
            
            print('Overall scores:')
            for i, _ in enumerate(self.model):
                    print(f'Model {i:2d} | score = {scores[i]:.5f}')
                    
        elif self.ms_criterion.startswith('nasi'):
            
            m = float(self.dataset.xs_train.shape[1])
            n = float(self.dataset.ys_train.shape[1])
            lr_inv = 1. / self.criterion_params.get('lr', 1.)
            factor = m * n * lr_inv
            
            xs = jnp.array(self.dataset.xs_train[alg.selected])
            
            print('Starting model selection with NASI')
            print(f'Trace norm upper bound = {factor}')
            
            scores = []
            best_score = -float('inf')
            best_score_idx = None
            best_score_nocomplex = -float('inf')
            best_score_idx_nocomplex = None
            
            for i, m in enumerate(self.model):
                
                K = m.get_ntk(x1=xs)
                tr_norm = jnp.linalg.norm(K, ord='nuc')
                
                if best_score < tr_norm <= factor:
                    best_score = tr_norm
                    best_score_idx = i
                
                if best_score_nocomplex < tr_norm:
                    best_score_nocomplex = tr_norm
                    best_score_idx_nocomplex = i
                    
                scores.append(tr_norm)
                print(f'Model {i:2d} | score = {tr_norm:.5f}')
        
            self.last_iter_data = {
                'scores': scores,
                'tr_ub': factor,
            }
            
        elif self.ms_criterion.startswith('naswot'):
            
            assert 'Torch' in str(type(self.model))
            
            scores = []
            inputs = torch.tensor(self.dataset.xs_train[alg.selected, :],
                                  dtype=torch.get_default_dtype(),
                                  device=self.model.device)
            
            for i, m in enumerate(self.model.models):
                
                net = m.model
                score = compute_naswot_sim(net=net, inputs=inputs)
                scores.append(score)
                print(f'Model {i:2d} | score = {score:.5f}')
                
            self.last_iter_data = dict()
        
        else:
            raise ValueError(f'Invalid ms_criterion: {self.ms_criterion}')

        # print(f'Ensemble score ({self.ensemble_scoring}) = {scores}')
        # self.model.reweight_models_log(scores)
        self.model.select_model(model_idx=np.argmax(scores))
        print(f'Best model = {np.argmax(scores)}')
        
        self.last_iter_data['scores'] = scores
        self.last_iter_data['best_model'] = np.argmax(scores)
        
    def _train_model(self):
            
        print(f'Training with {np.sum(self.selected)} points.')
        self.model.init_weights()
        
        for i, m in tqdm.tqdm(list(enumerate(self.model.models))):
            
            if not m.use_empirical_kernel:
                continue
            
            if 'Jax' in str(type(self.model)):
                    # TODO: add training for JAX fn as well
                raise ValueError
            
            elif 'Torch' in str(type(self.model)):
                done_flag = False
                for rnd in range(20):  # train and make sure don't get NaN value
                    m.init_weights()
                    torch_dataset, _ = data_loaders(
                        dataset=self.dataset, 
                        selected=self.get_selected(), 
                        batch_size=self.train_batch_sz, 
                        shuffle=True,
                        generate_reg_data=True,
                        in_class_val=self.in_class_val,
                        not_in_class_val=self.not_in_class_val,
                        device=m.device
                    )
                    train_torch(
                        model=m,
                        optimizer=self.optimiser_generator(m.parameters()),
                        criterion='mse',  # assume MSE loss to match the theory
                        train_loader=torch_dataset,
                        epochs=self.train_epochs,
                        progbar=False,
                    )
                    
                    m.eval()
                    input_ = torch.tensor(
                        np.asarray(self.dataset.xs_train[self.selected, :]), 
                        dtype=torch.get_default_dtype(), 
                        device=m.device
                    )
                    output = m(input_)
                    if not torch.isnan(output).any():
                        done_flag = True
                        break
                
                if not done_flag:
                    m.init_weights()
                    print(f'Model {i} - NaN value obtained in output in ALL rounds of training.')
                elif rnd > 0:
                    print(f'Model {i} - NaN value obtained in output in {rnd} rounds of training.')
                    
            else:
                assert False
