import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax

from .nn_model import JaxNNModel
from al_ntk.utils.kernels_helper import generate_batched_kernel, compute_kernel_in_batches, compute_diag_kernel_in_batches


class MLPJax(JaxNNModel):

    def __init__(self, in_dim=(1,), out_dim=1, hidden_count=1, width=1024, W_std=1., b_std=1., activation='relu',
                 activation_param=0., seed=None, parametrisation: str = 'ntk', 
                 use_empirical_kernel=False, kernel_batch_sz=128, dropout=None):
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.W_std = W_std
        self.b_std = b_std
        self.hidden_count = hidden_count
        self.width = width
        self.parametrisation = parametrisation
        self.kernel_batch_sz = kernel_batch_sz

        if activation == 'rbf':
            self.activation = lambda i: stax.Rbf(gamma=activation_param)
        elif activation == 'erf':
            self.activation = lambda i: stax.Erf()
        elif activation == 'sigmoid':
            self.activation = lambda i: stax.Sigmoid_like()
        elif activation == 'relu':
            self.activation = lambda i: stax.Relu()
        elif activation == 'relu_norm':
            self.activation = lambda i: stax.ABRelu(a=0., b=1.41421356237)
        elif activation == 'leaky_relu':
            self.activation = lambda i: stax.LeakyRelu(alpha=activation_param)
        elif activation == 'gelu':
            self.activation = lambda i: stax.Gelu()
        elif activation == 'sin':
            self.activation = lambda i: stax.Sin(a=1., b=activation_param, c=0.)
        elif activation == 'linear':
            self.activation = lambda i: stax.Monomial(1)
        else:
            raise ValueError(f'Invalid activation: {activation}')
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        for i in range(hidden_count):
            layers.extend([stax.Dense(width, W_std=W_std, b_std=b_std, parameterization=parametrisation),
                           self.activation(i)])
            if dropout is not None:
                layers.append(stax.Dropout(rate=dropout, mode='train'))
        layers.append(stax.Dense(out_dim, W_std=W_std, b_std=b_std, parameterization=parametrisation))
        self.layers = layers
        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(*layers)
        # self.apply_fn = jax.jit(self.apply_fn)
        # self.kernel_fn = jax.jit(
        #     fun=self.kernel_fn,
        #     static_argnames=['get']
        # )

        self.key = jax.random.PRNGKey(np.random.randint(1000000000) if seed is None else seed)
        self.params = None
        self.init_weights()

        self.use_empirical_kernel = use_empirical_kernel
        # check implementation parameter if get OOM errors
        self.emp_kernel_batched = generate_batched_kernel(
            kernel_fn=nt.empirical_kernel_fn(
                self.apply_fn, 
                implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
            ),
            batch_sz=self.kernel_batch_sz,
            returns_diagonal=False
        )
        # self.emp_kernel_batched = compute_kernel_in_batches(
        #     kernel_fn=nt.empirical_kernel_fn(
        #         self.apply_fn, 
        #         implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
        #     ),
        #     batch_sz=self.kernel_batch_sz,
        # )
        self.emp_kernel_diagonal_batched = generate_batched_kernel(
            kernel_fn=nt.empirical_kernel_fn(
                self.apply_fn, 
                diagonal_axes=(0,), 
                implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
            ),
            batch_sz=self.kernel_batch_sz,
            returns_diagonal=True
        )
        # self.emp_kernel_diagonal_batched = compute_diag_kernel_in_batches(
        #     kernel_fn=nt.empirical_kernel_fn(
        #         self.apply_fn, 
        #         diagonal_axes=(0,), 
        #         implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
        #     ),
        #     batch_sz=self.kernel_batch_sz,
        # )
        self.kernel_batched = generate_batched_kernel(
            kernel_fn=self.kernel_fn,
            batch_sz=self.kernel_batch_sz
        )
        # self.kernel_batched = compute_kernel_in_batches(
        #     kernel_fn=self.kernel_fn,
        #     batch_sz=self.kernel_batch_sz
        # )
        self.kernel_diagonal_batched = compute_diag_kernel_in_batches(
            kernel_fn=self.kernel_fn,
            batch_sz=self.kernel_batch_sz
        )
            

    def __call__(self, xs):
        return self.apply_fn(self.params, xs)

    def get_grad(self, input_x: np.ndarray, flatten: bool = True):
        assert self.out_dim == 1  # maybe fix later
        n = input_x.shape[0]
        g = jax.jacobian(self.apply_fn)(self.params, input_x)
        if flatten:
            return jnp.concatenate([w.reshape(n, -1) for layer in g for w in layer], axis=1)
        else:
            return g

    def _get_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray = None, 
                    get_diagonal_only: bool = False, kernel: str = 'ntk'):
        assert (not get_diagonal_only) or (x2 is None)
        # assert self.out_dim == 1  # have to fix later
        n1 = x1.shape[0]
        n2 = n1 if x2 is None else x2.shape[0]
        if self.use_empirical_kernel:
            if get_diagonal_only:
                return self.emp_kernel_diagonal_batched(x1=x1, x2=None, get=kernel, params=self.params)
            else:
                return self.emp_kernel_batched(x1=x1, x2=x2, get=kernel, params=self.params).reshape(n1, n2)
        else:
            if get_diagonal_only:
                return self.kernel_diagonal_batched(x1=x1, x2=x2, get=kernel)
            else:
                return self.kernel_batched(x1=x1, x2=x2, get=kernel).reshape(n1, n2)

    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='ntk')

    def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='nngp')
