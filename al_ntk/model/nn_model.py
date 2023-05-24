from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
import torch
from torch import nn as nn

from al_ntk.utils.kernels_torch import generate_ntk_kernel_torch, generate_batched_ntk_kernel_torch


class NNModel(ABC):

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def __call__(self, xs):
        raise NotImplementedError
    
    @abstractmethod
    def call_np(self, xs):
        raise NotImplementedError
    
    @abstractmethod
    def call_jnp(self, xs):
        raise NotImplementedError

    @abstractmethod
    def init_weights(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        """ Get NTK value

        Parameters
        ----------
        x1: input 1
        x2: input 2 (or set to None if want to compute kernel with x1 itself)
        get_diagonal_only: only return the diagonal of the matrix K(x1, x1)

        Returns
        -------

        """
        raise NotImplementedError


class TorchNNModel(torch.nn.Module, NNModel, ABC):

    def __init__(self, in_dim, out_dim, ntk_compute_method='ntk_vps', kernel_batch_sz=256, use_cuda=True, rand_idxs=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = None  # initialised by child class
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        self.ntk_compute_method = ntk_compute_method
        self.kernel_batch_sz = kernel_batch_sz
        self.rand_idxs = rand_idxs
        self._ntk = None  # initialised on first get_ntk call

    @abstractmethod
    def get_penultimate_and_final_output(self, xs):
        """ Function to return the penultimate output (before the final FC layer) and the final output
        Penultimate output needed for BADGE algorithm

        Parameters
        ----------
        xs

        Returns
        -------
        (penultimate output, final output)
        """
        raise NotImplementedError

    def init_weights(self):
        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # xavier initialises as normal with std = 2 / (in + out)
                # this is roughly matching with the 'standard' initialisation in neural-tangents
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
        self.model.apply(init_weights)
        self._ntk = None
        
    def __call__(self, xs):
        return self.model(xs)
    
    def call_np(self, xs):
        ys = self(torch.tensor(xs, dtype=torch.get_default_dtype(), device=self.device))
        return ys.numpy(force=True)
    
    def call_jnp(self, xs):
        ys = self.call_np(np.asarray(xs))
        return jnp.array(ys)
    
    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        
        # initialise kernel if haven't already
        if self._ntk is None:
            self._ntk = generate_batched_ntk_kernel_torch(
                model=self.model,
                batch_sz=self.kernel_batch_sz,
                method=self.ntk_compute_method,
                out_dim=self.out_dim,
                rand_idxs=self.rand_idxs
            )
        
        # convert x1 and x2 to correct types
        x1 = torch.tensor(np.asarray(x1), dtype=torch.get_default_dtype(), device=self.device)
        if x2 is not None:
            x2 = torch.tensor(np.asarray(x2), dtype=torch.get_default_dtype(), device=self.device)
            
        mat = self._ntk(
            x1=x1,
            x2=x2,
            get_diagonal_only=get_diagonal_only
        )
        return mat
    

class JaxNNModel(NNModel, ABC):

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.init_fn = None
        self.apply_fn = None
        self.params = None
        self.key = None
        
    def call_np(self, xs):
        ys = self(jnp.array(xs))
        return np.array(ys)
        
    def call_jnp(self, xs):
        return self(xs)

    @abstractmethod
    def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        raise NotImplementedError
    
    def init_weights(self):
        self.key, net_key = jax.random.split(self.key)
        _, self.params = self.init_fn(net_key, (-1, *self.in_dim))
