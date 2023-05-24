from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import jax
import jax.numpy as jnp
import torch
from torch import nn as nn

from .nn_model import TorchNNModel, JaxNNModel
from .mlp_torch import MLPTorch
from .mlp_jax import MLPJax
from .cnn_torch import CNNTorch


class EnsembleModelTorch(TorchNNModel):
    
    def __init__(self, models: List[TorchNNModel], model_init_weights: Union[int, np.ndarray] = None, 
                 min_weight_to_use: float = 1e-6):
        models = nn.ModuleList(models)
        in_dim = models[0].in_dim
        out_dim = models[0].out_dim
        use_cuda = models[0].use_cuda
        self.device = models[0].device
        
        for m in models:
            assert in_dim == m.in_dim
            assert out_dim == m.out_dim
            assert use_cuda == m.use_cuda
        
        super().__init__(
            in_dim=in_dim, 
            out_dim=out_dim, 
            ntk_compute_method=None, 
            kernel_batch_sz=None, 
            use_cuda=use_cuda, 
            rand_idxs=None
        )
        
        self.models = models
        self.min_weight_to_use = min_weight_to_use
        if isinstance(model_init_weights, int):
            self.select_model(model_idx=model_init_weights)
        else:
            self.reweight_models(new_weights=model_init_weights)
        
    @classmethod
    def construct_ensemble(cls, in_dim, out_dim, family: str, min_weight_to_use: float = 1e-6, 
                           use_empirical_kernel: bool = True, ntk_compute_method: str = 'jac_con', 
                           kernel_batch_sz: int = 256, rand_idxs: int = -1, 
                           use_cuda: bool = True, model_init_weights: Union[int, np.ndarray] = None, 
                           dropout_rate: List[float] = None,
                           mlp_activations: List[str] = None, mlp_width: List[int] = None, 
                           mlp_hidden_layers: List[int] = None, mlp_bias: List[float] = None, 
                           cnn_convs: List[List[int]] = None, cnn_hidden_layers: List[List[int]] = None, 
                           cnn_conv_kernel_sizes: List[List[int]] = None, cnn_conv_strides: List[List[int]] = None, 
                           cnn_pool_kernel_sizes: List[List[int]] = None,
                           parametrisation: str = 'ntk', generate_jax_model: bool = False):
        models = []
        
        # sorry about the nested loops
        if 'mlp' in family:
            for act in mlp_activations:
                if isinstance(act, str):
                    act_arg = None
                else:
                    act, act_arg = act
                for w in mlp_width:
                    for d in mlp_hidden_layers:
                        for b in mlp_bias:
                            for drp in dropout_rate:
                                models.append(MLPTorch(
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    hidden_count=d,
                                    width=w,
                                    W_std=1.,
                                    b_std=b,
                                    activation=act,
                                    activation_param=act_arg,
                                    dropout_rate=drp,
                                    ntk_compute_method=ntk_compute_method,
                                    kernel_batch_sz=kernel_batch_sz,
                                    use_cuda=use_cuda,
                                    rand_idxs=rand_idxs,
                                    use_empirical_kernel=use_empirical_kernel,
                                    parametrisation=parametrisation,
                                    generate_jax_model=generate_jax_model,
                                ))
        if 'cnn' in family:
            for h in cnn_hidden_layers:
                for drp in dropout_rate:
                    for conv in cnn_convs:
                        for ckv in cnn_conv_kernel_sizes:
                            for cs in cnn_conv_strides:
                                for pks in cnn_pool_kernel_sizes:
                                    models.append(CNNTorch(
                                        in_dim=in_dim,
                                        out_dim=out_dim,
                                        conv_layers=conv,
                                        hidden_layers=h,
                                        dropout_p=drp,
                                        conv_kernel_size=ckv, 
                                        conv_stride=cs, 
                                        pool_kernel_size=pks,
                                        ntk_compute_method=ntk_compute_method,
                                        kernel_batch_sz=kernel_batch_sz,
                                        use_cuda=use_cuda,
                                        rand_idxs=rand_idxs,
                                    ))
        if len(models) == 0:
            raise ValueError('Invalid family.')
        
        return EnsembleModelTorch(models=models, min_weight_to_use=min_weight_to_use, model_init_weights=model_init_weights)
    

    def reweight_models(self, new_weights=None):
        if new_weights is None:
            self.weights = torch.ones(size=(len(self.models),), 
                                      dtype=torch.get_default_dtype(), 
                                      device=self.models[0].device) / len(self.models)
        else:
            w = np.array(new_weights)
            w = w / np.sum(w)
            w = torch.tensor(w, dtype=torch.get_default_dtype(), device=self.device)
            self.weights = w
            
    def select_model(self, model_idx: int):
        w = np.eye(len(self.models))[model_idx]
        self.reweight_models(new_weights=w)
        
    def get_weights(self):
        return self.weights.detach().cpu().numpy()
    
    def __getitem__(self, key):
        return self.models[key]

    def init_weights(self):
        for m in self.models:
            m.init_weights()
        
    def __call__(self, xs):
        mats = []
        for (w, m) in zip(self.weights, self.models):
            if w > self.min_weight_to_use:
                mats.append(w * m(xs))
        return torch.sum(torch.stack(mats), dim=0)
    
    def get_penultimate_and_final_output(self, xs):
        raise ValueError('Invalid for ensemble models')
    
    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        mats = [float(w) * m.get_ntk(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only) 
            for (w, m) in zip(self.weights, self.models) if w > self.min_weight_to_use]
        return jnp.sum(jnp.stack(mats), axis=0)
        
        
class EnsembleModelJax(JaxNNModel):
    
    def __init__(self, models: List[JaxNNModel], model_init_weights: Union[int, np.ndarray] = None, 
                 min_weight_to_use: float = 1e-6):
        models = models
        in_dim = models[0].in_dim
        out_dim = models[0].out_dim
        
        for m in models:
            assert in_dim == m.in_dim
            assert out_dim == m.out_dim
        
        super().__init__(
            in_dim=in_dim, 
            out_dim=out_dim
        )
        
        self.models = models
        self.min_weight_to_use = min_weight_to_use
        if isinstance(model_init_weights, int):
            self.select_model(model_idx=model_init_weights)
        else:
            self.reweight_models(new_weights=model_init_weights)
        
    @classmethod
    def construct_ensemble(cls, in_dim, out_dim, family: str, min_weight_to_use: float = 1e-6, 
                           use_empirical_kernel: bool = True, kernel_batch_sz: int = 256, 
                           use_cuda: bool = True, model_init_weights: Union[int, np.ndarray] = None, 
                           dropout_rate: List[float] = None,
                           mlp_activations: List[str] = None, mlp_width: List[int] = None, 
                           mlp_hidden_layers: List[int] = None, mlp_bias: List[float] = None,
                           parametrisation: str = 'ntk'):
        models = []
        
        # sorry about the nested loops
        if 'mlp' in family:
            for act in mlp_activations:
                if isinstance(act, str):
                    act_arg = None
                else:
                    act, act_arg = act
                for w in mlp_width:
                    for d in mlp_hidden_layers:
                        for b in mlp_bias:
                            for drp in dropout_rate:
                                models.append(MLPJax(
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    hidden_count=d,
                                    width=w,
                                    W_std=1.,
                                    b_std=b,
                                    activation=act,
                                    activation_param=act_arg,
                                    kernel_batch_sz=kernel_batch_sz,
                                    use_empirical_kernel=use_empirical_kernel,
                                    parametrisation=parametrisation,
                                ))
                                
        if len(models) == 0:
            raise ValueError('Invalid family.')
        
        return EnsembleModelJax(models=models, min_weight_to_use=min_weight_to_use, model_init_weights=model_init_weights)
    

    def reweight_models(self, new_weights=None):
        if new_weights is None:
            self.weights = jnp.ones(shape=(len(self.models),))
        else:
            w = jnp.array(new_weights)
            w = w / jnp.sum(w)
            self.weights = w
            
    def get_weights(self):
        return np.array(self.weights)
        
    def select_model(self, model_idx: int):
        w = np.eye(len(self.models))[model_idx]
        self.reweight_models(new_weights=w)
    
    def __getitem__(self, key):
        return self.models[key]

    def init_weights(self):
        for m in self.models:
            m.init_weights()
        
    def __call__(self, xs):
        mats = []
        for (w, m) in zip(self.weights, self.models):
            if w > self.min_weight_to_use:
                mats.append(w * m(xs))
        return jnp.sum(jnp.stack(mats), axis=0)
    
    def get_penultimate_and_final_output(self, xs):
        raise ValueError('Invalid for ensemble models')
    
    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        mats = [float(w) * m.get_ntk(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only) 
            for (w, m) in zip(self.weights, self.models) if w > self.min_weight_to_use]
        return jnp.sum(jnp.stack(mats), axis=0)
    
    def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        mats = [float(w) * m.get_nngp(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only) 
                for (w, m) in zip(self.weights, self.models) if w > self.min_weight_to_use]
        return jnp.sum(jnp.stack(mats), axis=0)
        