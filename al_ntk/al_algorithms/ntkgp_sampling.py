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
from al_ntk.model.nn_model import JaxNNModel, TorchNNModel, NNModel
from .base_al_algorithm import ALAlgorithm
from al_ntk.gp import FullRankGP, FIC
from al_ntk.utils.entropy_helper import max_entropy_selector, max_entropy_selector_from_fn
from al_ntk.utils.kernels_helper import compute_kernel_in_batches, approximate_full_kernel_as_block_diag
from al_ntk.utils.maps import optim_map
from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.nn_training_torch import train_torch


@jax.jit
def criterion_ev(ys, mean, var):
    return - jnp.mean(var)


@jax.jit
def criterion_ev_w_labels(ys, mean, var):
    return - (jnp.mean((ys - mean) ** 2 + var[:, None]))


@jax.jit
def criterion_vol(ys, mean, var):
    return - jnp.mean(var ** 2)


@jax.jit
def criterion_vol_w_labels(ys, mean, var):
    return - (2. * jnp.mean(((ys - mean) ** 2 + var[:, None])**2 - (ys - mean) ** 4))


@jax.jit
def criterion_ent(ys, mean, cov):
    return - jnp.linalg.slogdet(cov)[1]


@jax.jit
def criterion_ent_masked(ys, mean, cov, mask):
    cov_transformed = mask[:, None] * cov
    cov_transformed = mask[:, None] * cov_transformed.T
    cov_transformed = cov_transformed + jnp.diag(1. - mask)
    return - jnp.linalg.slogdet(cov_transformed)[1]


@jax.jit
def criterion_mi(ys, mean, cov, mask, prior):
    cov_transformed = mask[:, None] * cov
    cov_transformed = mask[:, None] * cov_transformed.T
    cov_transformed = cov_transformed + jnp.diag(1. - mask)

    prior_transformed = mask[:, None] * prior
    prior_transformed = mask[:, None] * prior_transformed.T
    prior_transformed = prior_transformed + jnp.diag(1. - mask)
    return jnp.linalg.slogdet(prior_transformed)[1] - jnp.linalg.slogdet(cov_transformed)[1]


@jax.jit
def criterion_mv(ys, mean, var):
    return - jnp.max(var)


@jax.jit
def criterion_90v(ys, mean, var):
    return - jnp.percentile(var, q=90)


@jax.jit
def criterion_95v(ys, mean, var):
    return - jnp.percentile(var, q=95)


@jax.jit
def criterion_mlmoc(ys, mean, var, prev_mean):
    return jnp.mean(jnp.linalg.norm(mean - prev_mean, axis=1))


# tuple represents (need pseudolabels, only need gp variance)
CRITERIA = {
    'ev': (criterion_ev, False, True),
    'ev-w': (criterion_ev_w_labels, True, True),
    'vol': (criterion_vol, False, True),
    'vol-w': (criterion_vol_w_labels, True, True),
    'ent': (criterion_ent, False, False),
    'ent-m': (criterion_ent_masked, False, False),
    'mi': (criterion_mi, False, False),
    'mv': (criterion_mv, False, True),
    '90v': (criterion_90v, False, True),
    '95v': (criterion_95v, False, True),
    'mlmoc': (criterion_mlmoc, True, True),
}


class NTKGPApproxMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel = None,
                 criterion: Union[str, Callable] = 'ev', lazy_eval: bool = True,
                 posterior_approx_method: str = 'frgp', sigma_noise: float = 1e-3, reinit_gp: bool = False,
                 inducing_count: int = 100, inducing_pt_method: str = 'entropy', not_in_class_val: float = -1.,
                 use_train_set_for_test: bool = False, test_prop_seen_max: int = 5000, 
                 test_prop_method: str = 'random', keep_training_kernel_vals: bool = False, 
                 check_per_round: int = 6000, approx_kernel_as_block_diag: int = -1,
                 random_first_batch_count: int = None,
                 train: dict = None, optim: dict = None, use_cuda: bool = False):
        super().__init__(dataset=dataset, model=model)

        self.posterior_approx_method = posterior_approx_method
        self.lazy_eval = lazy_eval
        self.last_batch_idxs = []
        self.criterion_vals = []
        self.criterion_name = criterion
        self.initialised = False
        self.sigma_noise = sigma_noise
        self.not_in_class_val = not_in_class_val
        self.use_train_set_for_test = use_train_set_for_test
        self.test_prop_seen_max = test_prop_seen_max
        self.test_prop_method = test_prop_method
        self.reinit_gp = reinit_gp
        self.inducing_pt_method = inducing_pt_method
        self.inducing_count = inducing_count
        self.keep_training_kernel_vals = keep_training_kernel_vals
        self.check_per_round = check_per_round
        self.approx_kernel_as_block_diag = approx_kernel_as_block_diag
        self.random_first_batch_count = random_first_batch_count
        self.use_cuda = use_cuda
        
        if self.approx_kernel_as_block_diag < 0:
            self.kernel = self.model.get_ntk
        else:
            self.kernel = approximate_full_kernel_as_block_diag(
                kernel_fn=self.model.get_ntk,
                diag_batch_sz=self.approx_kernel_as_block_diag
            )

        if criterion not in CRITERIA.keys():
            raise ValueError(f'Invalid criterion')
        elif lazy_eval and (criterion in {'mlmoc'}):
            raise ValueError(f'{criterion} not valid with lazy_eval=True')
        else:
            self.criterion, self.use_pseudolabels, self.track_diag_only = CRITERIA[criterion]
        
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
        self._init_gp()
        
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
        
        if self.criterion_name in {'mlmoc'}:
            # only works if is TorchModelNN for now
            # TODO: make compatible with JAX (or include proper check to only allow torch models)
            prior_mean_fn = self._call_torch_model
        else:
            prior_mean_fn = None
        
        print('Initialising GP.')
        t = time.time()
        if self.posterior_approx_method == 'frgp':
            self.gp = FullRankGP(
                kernel_fn=self.kernel,
                Xn=self.xs_train,
                yn=jnp.array(self.ys_train),
                Xt=self.xs_test,
                sigma_noise=self.sigma_noise,
                keep_training_kernel_vals=self.keep_training_kernel_vals,
                only_track_diag=self.track_diag_only,
                use_train_set_for_test=self.val_set if self.val_set is not None else self.use_train_set_for_test,
                prior_mean_fn=prior_mean_fn
            )

        elif self.posterior_approx_method in {'sgp', 'sgpla'}:
            # for FIC don't need diagonals
            self.keep_training_kernel_vals = False
            self.gp = FIC(
                kernel_fn=self.kernel,
                Xn=self.xs_train,
                yn=jnp.array(self.ys_train),
                Xt=self.xs_test,
                yt=self.ys_test,
                sigma_noise=self.sigma_noise,
                inducing_pts=self.inducing_pt_method,
                m=self.inducing_count,
                keep_training_kernel_vals=self.keep_training_kernel_vals,
                only_track_diag=self.track_diag_only,
                use_train_set_for_test=self.val_set if self.val_set is not None else self.use_train_set_for_test
            )

        else:
            raise ValueError('Invalid posterior_approx_method')
        
        print(f'Done initialising GP, time = {time.time() - t}')

    def _update_pseudolabel(self):
        pseudolabels_tr, pseudolabels_te = self.dataset.generate_label_for_regression(
            train_mask=~self.selected,
            not_in_class_val=self.not_in_class_val)
        pseudolabels_tr = jnp.array(pseudolabels_tr)

        self.gp.update_labels(new_yn=pseudolabels_tr)
        
        if self.use_pseudolabels:

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
        
        else:
                        
            if self.use_train_set_for_test:
                self.ys_test = self.ys_train[self.val_set, :]
            else:
                self.ys_test = pseudolabels_te

    def preselect(self, idxs: np.ndarray):
        assert not self.selected.any()
        self.selected[idxs] = True
        self.gp.update_train(idxs)
        self._update_pseudolabel()

        if self.use_train_set_for_test and idxs.shape[0] > 0:
            if not self.used_test_subset:
                self.curr_mask = self.curr_mask.at[jnp.array(idxs)].set(0.)
            else:
                idxs_transformed = [self.val_set_map[x] for x in idxs if x in self.val_set_map.keys()]
                if len(idxs_transformed) > 0:
                    self.curr_mask = self.curr_mask.at[jnp.array(idxs_transformed)].set(0.)

        if self.lazy_eval:
            if self.track_diag_only:
                test_mean, test_var = self.gp.get_test_posterior_diagonal()
                self.last_best_score = self.criterion(ys=self.ys_test, mean=test_mean, var=test_var,
                                                      **self._get_criterion_kwargs())
            else:
                test_mean, test_cov = self.gp.get_test_posterior()
                self.last_best_score = self.criterion(ys=self.ys_test, mean=test_mean, cov=test_cov,
                                                      **self._get_criterion_kwargs())
        else:
            self.last_best_score = -np.inf

        self.initialised = True

    def get(self, n: int):

        if not self.initialised:
            
            # initialise everything properly
            self.preselect(jnp.arange(0))
            
        elif self.reinit_gp or self.do_training:
            
            if self.do_training and self.selected.any():
                # if train to get better NTK update, use model training with selected data
                self._train_model()
            else:
                # otherwise, just reinitialise the model
                self.model.init_weights()
            
            # if either did training or want to reinit model, then have to reset the GP
            self._init_gp()
            self.gp.update_train(np.argwhere(self.selected).flatten())
            self._update_pseudolabel()
            
            # also have to reset stats for lazy_init
            self.marginal_increase_ub = np.full(self.dataset.train_count(), fill_value=np.inf, dtype=np.float64)
            self.marginal_increase_ub[self.selected] = 0.
            if self.lazy_eval:
                if self.track_diag_only:
                    test_mean, test_var = self.gp.get_test_posterior_diagonal()
                    self.last_best_score = self.criterion(ys=self.ys_test, mean=test_mean, var=test_var,
                                                          **self._get_criterion_kwargs())
                else:
                    test_mean, test_cov = self.gp.get_test_posterior()
                    self.last_best_score = self.criterion(ys=self.ys_test, mean=test_mean, cov=test_cov,
                                                          **self._get_criterion_kwargs())
                    
        # store some intermediate data
        if self.criterion_name in {'mlmoc'}:
            self.round_mean = jnp.copy(self.gp.get_test_posterior_diagonal()[0])

        self.last_batch_idxs = []

        if self.random_first_batch_count and not self.selected.any():
            sz = min(n, self.random_first_batch_count)
            print(f'In first batch, will randomly select {sz} points out of {n} budget.')
            n = n - sz
            
            rand_idxs = np.random.choice(self.selected.shape[0], size=sz, replace=False)
            self.last_batch_idxs.extend(rand_idxs.tolist())
            self.selected[rand_idxs] = True
            if not self.reinit_gp:
                self._update_pseudolabel()
            
        if n < 1:
            # random batch took care of everything already
            pass
            
        elif self.posterior_approx_method in {'frgp', 'sgp'}:
            self._get_with_gp(n)

        elif self.posterior_approx_method == 'sgpla':
            self._get_with_sgp_linapprox(n)

        else:
            raise ValueError('Invalid method')

        if (n > 0) and (not self.reinit_gp):
            # if points were selected, do pseudolabel update
            self._update_pseudolabel()

    def _get_with_gp(self, n: int):
        
        # multiplier = self.check_per_round / (self.dataset.train_count() - np.count_nonzero(self.selected))
        # print(f'checked probability = {multiplier}')
        
        sigma2_I = self.sigma_noise**2 * jnp.eye(self.gp.K_tt_diag.shape[0])

        for i in tqdm.trange(n):

            best_score = -np.inf
            best_score_idx = None

            checked = 0

            for x in np.random.permutation(self.dataset.train_count()):

                if not self.selected[x]:
                    
                    if self.lazy_eval and (self.marginal_increase_ub[x] < best_score - self.last_best_score):
                        continue  # if marginal gain is already too low, ignore it
                    
                    checked += 1
                    if checked > self.check_per_round:
                        break
                    
                    if self.track_diag_only:
                        test_mean, test_var = self.gp.get_updated_test_posterior_diagonal(idxs=jnp.arange(x, x + 1))
                        score = self.criterion(ys=self.ys_test, mean=test_mean, var=test_var,
                                               **self._get_criterion_kwargs(new_i=x))
                    else:
                        test_mean, test_cov = self.gp.get_updated_test_posterior(idxs=jnp.arange(x, x + 1))
                        test_cov = test_cov + sigma2_I  # add diagonal term
                        score = self.criterion(ys=self.ys_test, mean=test_mean, cov=test_cov,
                                               **self._get_criterion_kwargs(new_i=x))
                    if score > best_score:
                        best_score_idx = x
                        best_score = score
                    if self.lazy_eval:
                        # update marginal decrease of criterion
                        self.marginal_increase_ub[x] = score - self.last_best_score
                    # print(x, score)
                        
            if best_score_idx is None:
                raise ValueError('Error with selection process')

            self.selected[best_score_idx] = True
            if self.lazy_eval:
                self.marginal_increase_ub[best_score_idx] = 0.
            self.last_best_score = best_score
            self.last_batch_idxs.append(best_score_idx)
            self.criterion_vals.append(best_score)
            self.gp.incrementally_update_train(jnp.array([best_score_idx]))
            # self.gp.update_train(jnp.argwhere(self.selected).flatten())
            self.curr_mask = self._get_updated_mask(best_score_idx)

    def _get_with_sgp_linapprox(self, n: int):

        for i in tqdm.trange(n):

            if self.track_diag_only:
                func=lambda ys, mean, var: self.criterion(ys=ys, mean=mean, var=var,
                                                          **self._get_criterion_kwargs())
            else:
                func=lambda ys, mean, cov: self.criterion(ys=ys, mean=mean, cov=cov,
                                                          **self._get_criterion_kwargs())
                
            scores = self.gp.get_change_fn_given_added_element(func=func)
            scores = scores.at[self.selected].set(-jnp.inf)
            best_score_idx = int(jnp.argmax(scores))

            self.last_best_score = scores[best_score_idx]
            self.selected[best_score_idx] = True
            self.last_batch_idxs.append(best_score_idx)
            self.criterion_vals.append(scores[best_score_idx])
            self.gp.incrementally_update_train(jnp.array([best_score_idx]))
            self.curr_mask = self._get_updated_mask(best_score_idx)
            # self.gp.update_train(jnp.argwhere(self.selected).flatten())
            
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

    def _get_updated_mask(self, new_i):
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

    def _get_criterion_kwargs(self, new_i=None):
        extra_kwargs = dict()
        if self.criterion_name == 'mi':
            extra_kwargs['prior'] = self.gp.K_tt + self.sigma_noise ** 2 * jnp.eye(self.gp.K_tt.shape[0])
            
        if self.criterion_name in {'ent-m', 'mi'}:
            if new_i is None:
                extra_kwargs['mask'] = self.curr_mask
            else:
                extra_kwargs['mask'] = self._get_updated_mask(new_i=new_i)
                
        if self.criterion_name in {'mlmoc'}:
            # the mean of NN before adding an extra point in
            extra_kwargs['prev_mean'] = self.round_mean
            
        return extra_kwargs
