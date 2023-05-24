import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.special
from torch import optim
import jax.numpy as jnp

from .nn_model import TorchNNModel
from .mlp_jax import MLPJax
from .components.lambda_layer_torch import LambdaLayer


SQRT_2 = 1.41421356237
HALF_PI = 1.57079632679
QUAR_PI = 0.78539816339


class LinearNTKParam(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 W_std: float = 1., b_std: float = 1.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_std = W_std
        self.b_std = b_std

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.W_scaling = W_std / (self.in_features ** 0.5)
        self.b_scaling = b_std
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight.data, std=1.)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias.data, std=1.)

    def forward(self, x):
        return F.linear(x, self.W_scaling * self.weight, self.b_scaling * self.bias)


class MLPTorch(TorchNNModel):

    def __init__(self, in_dim=(1,), out_dim=1, hidden_count=1, width=1024, W_std=1., b_std=1., activation='relu', parametrisation: str = 'standard',
                 activation_param=0., seed=None, dropout_rate=None, ntk_compute_method='ntk_vps', kernel_batch_sz=256, 
                 use_empirical_kernel=True, use_cuda=True, rand_idxs=-1, generate_jax_model=False):
        super().__init__(in_dim=in_dim, out_dim=out_dim, use_cuda=use_cuda,
                         ntk_compute_method=ntk_compute_method, kernel_batch_sz=kernel_batch_sz, rand_idxs=rand_idxs)
        self.W_std = W_std
        self.b_std = b_std
        self.width = width
        self.dropout_rate = dropout_rate
        self.use_empirical_kernel = use_empirical_kernel
        self.parametrisation = parametrisation

        if seed is not None:
            torch.manual_seed(seed)

        if activation == 'sigmoid':
            self.activation = lambda: nn.Sigmoid()
        elif activation == 'relu':
            self.activation = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = lambda: nn.LeakyReLU(negative_slope=activation_param)
        elif activation == 'gelu':
            self.activation = lambda: nn.GELU()
        elif activation == 'erf':
            self.activation = lambda: LambdaLayer(fn=lambda xs: torch.special.erf(xs))
        elif activation == 'tanh':
            self.activation = lambda: nn.Tanh()
        elif activation == 'sin':
            self.activation = lambda: LambdaLayer(fn=lambda xs: torch.sin(activation_param * xs))
        elif activation == 'cos':
            self.activation = lambda: LambdaLayer(fn=lambda xs: torch.sin(activation_param * xs + HALF_PI))
        elif activation == 'rbf':
            factor = (2. * activation_param) ** 0.5
            # mathcing RBF from JAX
            self.activation = lambda: LambdaLayer(fn=lambda xs: SQRT_2 * torch.sin(factor * xs + QUAR_PI))
        else:
            raise ValueError(f'Invalid activation: {activation}')

        layers = []
        in_width = in_dim[0]
        for _ in range(hidden_count):
            if parametrisation == 'standard':
                layers.extend([nn.Linear(in_features=in_width, out_features=width, bias=(b_std is not None)),
                               self.activation()])
            elif parametrisation == 'ntk':
                layers.extend([LinearNTKParam(in_features=in_width, out_features=width, bias=(b_std is not None), W_std=W_std, b_std=b_std),
                               self.activation()])
            else:
                raise ValueError('Invalid parametrisation')
            if self.dropout_rate is not None:
                layers.append(nn.Dropout(p=self.dropout_rate))
            in_width = width
        if parametrisation == 'standard':
            layers.append(nn.Linear(in_features=in_width, out_features=out_dim, bias=(b_std is not None)))
        elif parametrisation == 'ntk':
            layers.append(LinearNTKParam(in_features=in_width, out_features=out_dim, bias=(b_std is not None), W_std=W_std, b_std=b_std))
        else:
            raise ValueError('Invalid parametrisation')

        self.model = nn.Sequential(*layers)
        if self.use_cuda:
            # if torch.cuda.device_count() > 1:
            #     print(f'Running model on {torch.cuda.device_count()} devices')
            #     self.model = nn.DataParallel(self.model)
            self.model.cuda()
        self.init_weights()
        
        if not self.use_empirical_kernel or generate_jax_model:
            # for using theoretical NTK
            # interally construct the JAX model with the exact same parameters
            self.equiv_jax_model = MLPJax(
                in_dim=in_dim, out_dim=out_dim,
                hidden_count=hidden_count, width=self.width, W_std=W_std, b_std=b_std, 
                activation=activation, activation_param=activation_param,
                use_empirical_kernel=False, kernel_batch_sz=kernel_batch_sz,
                parametrisation=self.parametrisation
            )
        else:
            self.equiv_jax_model = None
            
    def init_weights(self):
        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # xavier initialises as normal with std = 2 / (in + out)
                # this is roughly matching with the 'standard' initialisation in neural-tangents
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, LinearNTKParam):
                torch.nn.init.normal_(m.weight.data, std=1.)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data, std=1.)
        self.model.apply(init_weights)
        self._ntk = None

    def get_penultimate_and_final_output(self, xs):
        if self.dropout_rate is not None:
            penultimate = self.model[:-3](xs)
            final = self.model[-3:](penultimate)
        else:
            penultimate = self.model[:-2](xs)
            final = self.model[-2:](penultimate)
        return penultimate, final
    
    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        if self.use_empirical_kernel:
            return super().get_ntk(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only)
        else:
            return self.equiv_jax_model.get_ntk(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only)
        
    def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        if self.use_empirical_kernel:
            raise NotImplementedError
        else:
            return self.equiv_jax_model.get_nngp(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only)
