from typing import Callable, Union
from functools import partial

import numpy as np
import jax
from jax import random
import torch
import jax.numpy as jnp
import tqdm
import time
from sklearn.cluster import kmeans_plusplus

from al_ntk.dataset import Dataset
from al_ntk.model import JaxNNModel, TorchNNModel, NNModel
from .base_al_algorithm import ALAlgorithm
from al_ntk.gp import FullRankGP, FIC
from al_ntk.utils.entropy_helper import max_entropy_selector, max_entropy_selector_from_fn
from al_ntk.utils.kernels_helper import compute_kernel_in_batches, approximate_full_kernel_as_block_diag
from al_ntk.utils.maps import optim_map
from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.nn_training_torch import train_torch


@jax.jit
def criterion_mlmoc(prev_mean, mean, mask):
    return jnp.mean(mask * jnp.linalg.norm(mean - prev_mean, axis=1))


class MLMOCMethodAlt(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel = None,
                 sigma_noise: float = 1e-3, not_in_class_val: float = -1.,
                 use_train_set_for_test: bool = True, test_prop_seen_max: int = 50000, 
                 test_prop_method: str = 'kmpp', keep_training_kernel_vals: bool = False, 
                 approx_kernel_as_block_diag: int = -1,
                 first_batch_random_count: int = 0,
                 train: dict = None, optim: dict = None, use_cuda: bool = False):
        super().__init__(dataset=dataset, model=model)

        self.last_batch_idxs = []
        self.criterion_vals = []
        self.initialised = False
        self.sigma_noise = sigma_noise
        self.not_in_class_val = not_in_class_val
        self.use_train_set_for_test = use_train_set_for_test
        self.test_prop_seen_max = test_prop_seen_max
        self.test_prop_method = test_prop_method
        self.keep_training_kernel_vals = keep_training_kernel_vals
        self.approx_kernel_as_block_diag = approx_kernel_as_block_diag
        self.first_batch_random_count = first_batch_random_count
        self.use_cuda = use_cuda
        
        if self.approx_kernel_as_block_diag < 0:
            self.kernel = self.model.get_ntk
        else:
            self.kernel = approximate_full_kernel_as_block_diag(
                kernel_fn=self.model.get_ntk,
                diag_batch_sz=self.approx_kernel_as_block_diag
            )
        
        if train is not None or optim is not None:
            self.do_training = True
            self.train_epochs = train['epochs']
            self.train_batch_sz = train['data_loader_args']['batch_size']
            self.optimiser_generator = lambda params: optim_map[optim['name']](params, **optim['params'])
            self.in_class_val = train['data_loader_args']['in_class_val']
            self.not_in_class_val = train['data_loader_args']['not_in_class_val']
        else:
            self.do_training = False

        self.xs_train = jnp.array(self.dataset.xs_train)
        self.ys_train = jnp.zeros(shape=(self.dataset.train_count(), self.dataset.output_dimensions()))
        self.xs_test = self.xs_train if use_train_set_for_test else jnp.array(self.dataset.xs_test)
        self.ys_test = self.ys_train if use_train_set_for_test else jnp.zeros(shape=(self.dataset.test_count(), self.dataset.output_dimensions()))
        
        self.used_test_subset = self.test_prop_seen_max < self.xs_test.shape[0]
        if self.use_train_set_for_test and self.used_test_subset:
            self._select_sub_test_set()
            self.val_set = jnp.sort(self.val_set)
            self.xs_test = self.xs_test[jnp.array(self.val_set), :]
            self.ys_test = self.ys_test[jnp.array(self.val_set), :]
            self.val_set_map = {x: i for (i, x) in enumerate(np.asarray(self.val_set).tolist())}
        else:
            self.val_set = None
            self.val_set_map = None

        self.curr_mask = jnp.ones(self.xs_test.shape[0])
        self.marginal_increase_ub = np.full(self.dataset.train_count(), fill_value=np.inf, dtype=np.float64)
        self.last_best_score = None

        self.gp = None
        
    def _select_sub_test_set(self):
        if self.test_prop_method == 'random':
            rand = np.random.RandomState(42)
            self.val_set = jnp.array(rand.permutation(self.xs_test.shape[0])[:self.test_prop_seen_max])
        elif self.test_prop_method == 'entropy':
            self.val_set = max_entropy_selector_from_fn(
                kernel_fn=self.kernel,
                xs=self.xs_test,
                m=self.test_prop_seen_max,
                entropy_lb=-10.
            )
            self.val_set = jnp.array(self.val_set)
        elif self.test_prop_method == 'mixed':
            # first pick some high entropy points
            val_set = max_entropy_selector_from_fn(
                kernel_fn=self.kernel,
                xs=self.xs_test,
                m=self.test_prop_seen_max,
                entropy_lb=-10.
            )
            # if quota remains, pick random points to fill
            if len(val_set) < self.test_prop_seen_max:
                rand = np.random.RandomState(42)
                remaining = list(set(range(self.xs_test.shape[0])).difference(val_set))
                val_set_extra = rand.permutation(remaining)[:(self.test_prop_seen_max - len(val_set))]
                val_set += list(val_set_extra)
                val_set.sort()
            self.val_set = jnp.array(val_set)
        elif self.test_prop_method == 'kmpp':
            xs_flatten = self.xs_test.reshape((self.xs_test.shape[0], -1))
            _, self.val_set = kmeans_plusplus(np.asarray(xs_flatten), n_clusters=self.test_prop_seen_max)
            self.val_set = jnp.array(self.val_set)
        elif self.test_prop_method == 'mixed-kmpp':
            xs_flatten = self.xs_test.reshape((self.xs_test.shape[0], -1))
            _, val_set = kmeans_plusplus(np.asarray(xs_flatten), n_clusters=self.test_prop_seen_max // 2, random_state=42)
            val_set = val_set.tolist()
            rand = np.random.RandomState(42)
            remaining = list(set(range(self.xs_test.shape[0])).difference(val_set))
            val_set_extra = rand.permutation(remaining)[:(self.test_prop_seen_max // 2)]
            val_set += list(val_set_extra)
            val_set.sort()
            self.val_set = jnp.array(val_set)
        elif self.test_prop_method == 'kmpp-cos':
            xs_flatten = self.xs_test.reshape((self.xs_test.shape[0], -1))
            xs_flatten = xs_flatten / jnp.linalg.norm(xs_flatten, axis=1)[:, None]
            _, self.val_set = kmeans_plusplus(np.asarray(xs_flatten), n_clusters=self.test_prop_seen_max)
            self.val_set = jnp.array(self.val_set)
        else:
            raise ValueError('Invalid test_prop_method')
        
    def _call_torch_model(self, xs):
        input_ = torch.tensor(
            np.asarray(xs), 
            dtype=torch.get_default_dtype(), 
            device=self.model.device
        )
        output = self.model(input_).detach().cpu().numpy()
        return jnp.array(output)

    def _init_gp(self):
        
        # only works if is TorchModelNN for now
        # TODO: make compatible with JAX (or include proper check to only allow torch models)
        prior_mean_fn = self._call_torch_model
        
        print('Initialising GP.')
        t = time.time()
        
        self.gp = FullRankGP(
            kernel_fn=self.kernel,
            Xn=self.xs_train,
            yn=jnp.array(self.ys_train),
            Xt=self.xs_test,
            sigma_noise=self.sigma_noise,
            keep_training_kernel_vals=self.keep_training_kernel_vals,
            only_track_diag=True,
            use_train_set_for_test=self.val_set if self.val_set is not None else self.use_train_set_for_test,
            prior_mean_fn=prior_mean_fn
        )
        
        print(f'Done initialising GP, time = {time.time() - t}')

    def _update_pseudolabel(self):
        pseudolabels_tr, pseudolabels_te = self.dataset.generate_label_for_regression(
            train_mask=~self.selected,
            not_in_class_val=self.not_in_class_val)
        pseudolabels_tr = jnp.array(pseudolabels_tr)

        self.gp.update_labels(new_yn=pseudolabels_tr)

        if self.do_training:
            # if do training, then the pseudolabel comes from the actual trained model
            # restrict this to only Torch models
            train_mean = self._call_torch_model(self.xs_train)
            test_mean = self._call_torch_model(self.xs_test)
        else:
            # otherwise, pseudolabels come from the GP prediction
            train_mean = self.gp.get_train_posterior_mean()
            test_mean, _ = self.gp.get_test_posterior_diagonal()
        
        if self.dataset.problem_type == 'classification':
            embedding = self.not_in_class_val + (1. - self.not_in_class_val) * jnp.eye(self.dataset.output_dimensions())
            train_mean = embedding[jnp.argmax(train_mean, axis=1), :]
            test_mean = embedding[jnp.argmax(train_mean, axis=1), :]
        elif self.dataset.problem_type == 'binary_classification':
            train_mean = self.not_in_class_val + (1. - self.not_in_class_val) * (train_mean > 0)
            test_mean = self.not_in_class_val + (1. - self.not_in_class_val) * (train_mean > 0)

        if self.do_training:
            # if do training, assume we are also using the true labels
            # so get the true label based on which item is selected already
            unlabelled = ~jnp.array(self.selected)
            self.ys_train = pseudolabels_tr.at[unlabelled].set(train_mean[unlabelled, :])
        else:
            # if no training, just use the pseudolabels for selection
            self.ys_train = pseudolabels_tr
        
        if self.use_train_set_for_test:
            # if train set and test set are the same, use it
            self.ys_test = self.ys_train[self.val_set, :]
        else:
            # otherwise, just use pseudo-test mean
            self.ys_test = test_mean

        self.gp.update_labels(new_yn=self.ys_train, new_yt=self.ys_test)

    def preselect(self, idxs: np.ndarray):
        assert not self.selected.any()
        self.selected[idxs] = True
        
        if self.gp is None:
            self._init_gp()
        self.gp.update_train(idxs)
        self._update_pseudolabel()

        if self.use_train_set_for_test and idxs.shape[0] > 0:
            if not self.used_test_subset:
                self.curr_mask = self.curr_mask.at[jnp.array(idxs)].set(0.)
            else:
                idxs_transformed = [self.val_set_map[x] for x in idxs if x in self.val_set_map.keys()]
                if len(idxs_transformed) > 0:
                    self.curr_mask = self.curr_mask.at[jnp.array(idxs_transformed)].set(0.)

        self.initialised = True

    def get(self, n: int):

        if not self.initialised:
            
            if self.first_batch_random_count > 0:
                m = min(self.first_batch_random_count, n)
                n = n - m
                rand_batch = np.random.choice(a=self.dataset.train_count(), 
                                              size=m, 
                                              replace=False)
                self.preselect(rand_batch)
                print(f'First batch preselect {self.first_batch_random_count} random points.')
                
            else:
                # initialise everything properly
                self.preselect(np.arange(0))
            
        else:
            
            if self.selected.any():
                # if train to get better NTK update, use model training with selected data
                self._train_model()
            else:
                # otherwise, just reinitialise the model
                self.model.init_weights()
            
            # if either did training or want to reinit model, then have to reset the GP
            self._init_gp()
            self.gp.update_train(np.argwhere(self.selected).flatten())
            self._update_pseudolabel()
            
        if n <= 0:
            return
                    
        # store some intermediate data
        round_mean = jnp.copy(self.gp.get_test_posterior_diagonal()[0])

        self.last_batch_idxs = []

        scores = np.empty(shape=(self.dataset.train_count(),))

        for x in tqdm.trange(self.dataset.train_count()):

            if self.selected[x]:
                scores[x] = -np.inf
            
            else:    
                test_mean, test_var = self.gp.get_updated_test_posterior_diagonal(idxs=jnp.arange(x, x + 1))
                scores[x] = criterion_mlmoc(
                    prev_mean=round_mean,
                    mean=test_mean, 
                    mask=self._get_updated_mask(new_i=x)
                )

        best_idxs = np.argsort(scores)[-n:]
        self.selected[best_idxs] = True
            
    def _train_model(self):
        
        print(f'Training with {np.sum(self.selected)} points.')
        
        # THIS IS BAD TYPE CHECKING - CHANGE IT LATER
        # too lazy to fix Python relative imports
        if 'Jax' in str(type(self.model)):
            # TODO: add training for JAX fn as well
            raise ValueError
        elif 'Torch' in str(type(self.model)):
            
            done_flag = False
            for rnd in range(20):  # train and make sure don't get NaN value
                self.model.init_weights()
                torch_dataset, _ = data_loaders(
                    dataset=self.dataset, 
                    selected=self.get_selected(), 
                    batch_size=self.train_batch_sz, 
                    shuffle=True,
                    generate_reg_data=True,
                    in_class_val=self.in_class_val,
                    not_in_class_val=self.not_in_class_val,
                    device=self.model.device
                )
                train_torch(
                    model=self.model,
                    optimizer=self.optimiser_generator(self.model.parameters()),
                    criterion='mse',  # assume MSE loss to match the theory
                    train_loader=torch_dataset,
                    epochs=self.train_epochs,
                    clip_value=10.,  # otherwise seems to always give nan
                    progbar=True,
                )
                
                if not jnp.isnan(self._call_torch_model(self.xs_train)).any():
                    done_flag = True
                    break
                else:
                    print(f'NaN value obtained in output from round {rnd} of training.')
            
            if not done_flag:
                raise ValueError('Error with training process - always getting NaN')
            
        else:
            assert False

    def _get_updated_mask(self, new_i=None):
        if new_i is None:
            return self.curr_mask
        
        if self.use_train_set_for_test:
            if self.used_test_subset:
                if new_i in self.val_set_map.keys():
                    return self.curr_mask.at[self.val_set_map[new_i]].set(0.)
                else:
                    return self.curr_mask
            else:
                return self.curr_mask.at[new_i].set(0.)
            
        else:
            return self.curr_mask
