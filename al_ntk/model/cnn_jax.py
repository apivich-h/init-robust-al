from functools import partial

import numpy as np
import jax
from jax.example_libraries import stax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax as nt_stax

from .nn_model import JaxNNModel
from al_ntk.utils.kernels_helper import generate_batched_kernel, compute_diag_kernel_in_batches


def DenseNTK(out_dim, W_std=1., b_std=None):
    # dense layer but with NTK parametrisation
    init_fn, apply_fn, _ = nt_stax.Dense(out_dim=out_dim, W_std=W_std, b_std=b_std)
    return init_fn, apply_fn


class CNNJax(JaxNNModel):

    def __init__(self, in_dim=(1, 28, 28), out_dim=10,
                 conv_layers=(32, 64), hidden_layers=(512,), dropout_p=0.5,
                 conv_kernel_size=(5, 5), conv_stride=(1, 1), pool_kernel_size=(2, 2),
                 W_std=1., b_std=None,
                 seed=None, use_empirical_kernel=True, kernel_batch_sz=128):
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        
        in_dim = tuple(in_dim)
        conv_kernel_size = tuple(conv_kernel_size)
        conv_stride = tuple(conv_stride)
        pool_kernel_size = tuple(pool_kernel_size)
        self.true_in_dim = (*in_dim[1:], in_dim[0])
        
        layers = []
        for c in conv_layers:
            layers.extend([
                nt_stax.Conv(out_chan=c, filter_shape=conv_kernel_size, strides=conv_stride),
                # nt_stax.Dropout(rate=dropout_p),
                nt_stax.AvgPool(window_shape=pool_kernel_size, strides=pool_kernel_size),
                nt_stax.Relu()
            ])
        layers.append(nt_stax.Flatten())
        for h in hidden_layers:
            layers.extend([
                nt_stax.Dense(out_dim=h, W_std=W_std, b_std=b_std),
                # nt_stax.Dropout(p=dropout_p),
                nt_stax.Relu(),
            ])
        layers.append(nt_stax.Dense(out_dim=self.out_dim, W_std=W_std, b_std=b_std))

        # layers = [
        #     nt_stax.Conv(out_chan=32, filter_shape=(5, 5), strides=(1, 1)),
        #     nt_stax.Relu(),  # swap ordering from actual a bit
        #     # nt_stax.Dropout(rate=dropout_p),
        #     # nt_stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
        #     nt_stax.Conv(out_chan=64, filter_shape=(5, 5), strides=(1, 1)),
        #     nt_stax.Relu(),  # swap ordering from actual a bit
        #     # nt_stax.Dropout(rate=dropout_p),
        #     # nt_stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
        #     nt_stax.Flatten(),
        #     nt_stax.Dense(out_dim=hidden_sz, W_std=W_std, b_std=b_std),
        #     nt_stax.Relu(),
        #     # nt_stax.Dropout(rate=dropout_p),
        #     nt_stax.Dense(out_dim=self.out_dim, W_std=W_std, b_std=b_std)
        #  ]

        self.init_fn, self.apply_fn, self.kernel_fn = nt_stax.serial(*layers)

        self.key = jax.random.PRNGKey(np.random.randint(1000000000) if seed is None else seed)
        self.params = None
        self.init_weights()

        self.key, self.nn_key = jax.random.split(self.key)

        self.kernel_batch_sz = kernel_batch_sz
        self.use_empirical_kernel = use_empirical_kernel
        # check implementation parameter if get OOM errors
        if use_empirical_kernel:
            self.emp_kernel_batched = generate_batched_kernel(
                kernel_fn=nt.empirical_kernel_fn(
                    self.apply_fn, 
                    implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
                ),
                batch_sz=self.kernel_batch_sz,
                returns_diagonal=False
            )
            # self.emp_kernel_diagonal_batched = generate_batched_kernel(
            #     kernel_fn=nt.empirical_kernel_fn(
            #         self.apply_fn, 
            #         diagonal_axes=(0,), 
            #         implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
            #     ),
            #     batch_sz=self.kernel_batch_sz,
            #     returns_diagonal=True
            # )
            self.emp_kernel_diagonal_batched = compute_diag_kernel_in_batches(
                kernel_fn=nt.empirical_kernel_fn(
                    self.apply_fn, 
                    implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
                ),
                batch_sz=self.kernel_batch_sz,
            )
        else:
            self.emp_kernel_batched = None
            self.emp_kernel_diagonal_batched = None
            self.kernel_batched = generate_batched_kernel(
                kernel_fn=self.kernel_fn,
                batch_sz=self.kernel_batch_sz
            )
            self.kernel_diagonal_batched = compute_diag_kernel_in_batches(
                kernel_fn=self.kernel_fn,
                batch_sz=self.kernel_batch_sz,
            )
        
    def init_weights(self):
        self.key, net_key = jax.random.split(self.key)
        _, self.params = self.init_fn(net_key, (-1, *self.true_in_dim))

    def __call__(self, xs):
        self.nn_key, key = jax.random.split(self.nn_key)
        # for jax the channel is the last dim unlike in pytorch
        return self.apply_fn(self.params, jnp.moveaxis(xs, 1, -1), rng=key)

    def _get_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False, kernel: str = 'ntk'):
        assert (not get_diagonal_only) or (x2 is None)
        x1t = jnp.moveaxis(x1, 1, -1)
        x2t = None if x2 is None else jnp.moveaxis(x2, 1, -1)
        n1 = x1.shape[0]
        n2 = n1 if x2 is None else x2.shape[0]
        if self.use_empirical_kernel:
            if get_diagonal_only:
                return self.emp_kernel_diagonal_batched(x1=x1t, x2=None, get=kernel, params=self.params)
            else:
                return self.emp_kernel_batched(x1=x1t, x2=x2t, get=kernel, params=self.params).reshape(n1, n2)
        else:
            if get_diagonal_only:
                return self.kernel_diagonal_batched(x1=x1t, x2=None, get=kernel)
            else:
                return self.kernel_batched(x1=x1t, x2=x2t, get=kernel).reshape(n1, n2)

    def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='ntk')

    def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
        return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='nngp')

# class MNISTModelJax(JaxNNModel):

#     def __init__(self, in_dim=(1, 28, 28), out_dim=10, hidden_sz=128, W_std=1., b_std=None, dropout_p=0.5,
#                  seed=None, use_empirical_kernel=True, only_use_fc_for_kernel=True):
#         super().__init__(in_dim=in_dim, out_dim=out_dim)
#         assert use_empirical_kernel or only_use_fc_for_kernel
        
#         self.true_in_dim = (*in_dim[1:], in_dim[0])

#         layers = [
#             stax.Conv(out_chan=32, filter_shape=(5, 5), strides=(1, 1)),
#             stax.Dropout(rate=dropout_p),
#             stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
#             stax.Relu,
#             stax.Conv(out_chan=64, filter_shape=(5, 5), strides=(1, 1)),
#             stax.Dropout(rate=dropout_p),
#             stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
#             stax.Relu,
#             stax.Flatten,
#             DenseNTK(out_dim=hidden_sz, W_std=W_std, b_std=b_std),
#             stax.Relu,
#             stax.Dropout(rate=dropout_p),
#             DenseNTK(out_dim=self.out_dim, W_std=W_std, b_std=b_std)
#          ]

#         self.init_fn, self.apply_fn = stax.serial(*layers)

#         self.key = jax.random.PRNGKey(np.random.randint(1000000000) if seed is None else seed)
#         self.params = None
#         self.init_weights()

#         self.key, self.nn_key = jax.random.split(self.key)

#         self.use_empirical_kernel = use_empirical_kernel
#         self.only_use_fc_for_kernel = only_use_fc_for_kernel
#         self.emp_kernel = jax.jit(
#             fun=nt.empirical_kernel_fn(partial(self.apply_fn, rng=self.nn_key),
#                                        implementation=3),
#             static_argnames=['get']
#         )
#         self.emp_kernel_diagonal = jax.jit(
#             fun=nt.empirical_kernel_fn(partial(self.apply_fn, rng=self.nn_key),
#                                        diagonal_axes=(0,), implementation=3),
#             static_argnames=['get']
#         )
        
#     def init_weights(self):
#         self.key, net_key = jax.random.split(self.key)
#         _, self.params = self.init_fn(net_key, (-1, *self.true_in_dim))

#     def __call__(self, xs):
#         self.nn_key, key = jax.random.split(self.nn_key)
#         # for jax the channel is the last dim unlike in pytorch
#         return self.apply_fn(self.params, jnp.moveaxis(xs, 1, -1), rng=key)

#     def _get_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False, kernel: str = 'ntk'):
#         assert (not get_diagonal_only) or (x2 is None)
#         x1t = jnp.moveaxis(x1, 1, -1)
#         x2t = None if x2 is None else jnp.moveaxis(x2, 1, -1)
#         if self.use_empirical_kernel:
#             n1 = x1.shape[0]
#             n2 = n1 if x2 is None else x2.shape[0]
#             if get_diagonal_only:
#                 return self.emp_kernel_diagonal(x1=x1t, x2=None, get=kernel, params=self.params)
#             else:
#                 return self.emp_kernel(x1=x1t, x2=x2t, get=kernel, params=self.params).reshape(n1, n2)
#         else:
#             assert False, "You shouldn't be here...?"

#     def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
#         return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='ntk')

#     def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
#         return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='nngp')

# class MNISTModelJAXSegmented(NNModel):
#     """
#     Same as MNISTModel but convolution and FC layer split
#     so that the NTK is computed only wrt the FC layer parameters
#     """
#     def __init__(self, in_dim=(28, 28, 1), out_classes=10, hidden_sz=128, W_std=1., b_std=None, dropout_p=0.5,
#                  seed=None, use_empirical_kernel=False):
#         super().__init__()
#         self.in_dim = in_dim

#         conv_layers = [
#             stax.Conv(out_chan=32, filter_shape=(5, 5), strides=(1, 1)),
#             stax.Dropout(rate=dropout_p),
#             stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
#             stax.Relu,
#             stax.Conv(out_chan=64, filter_shape=(5, 5), strides=(1, 1)),
#             stax.Dropout(rate=dropout_p),
#             stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
#             stax.Relu,
#             stax.Flatten
#         ]

#         fc_layers = [
#             nt_stax.Dense(out_dim=hidden_sz, W_std=W_std, b_std=b_std),
#             nt_stax.Relu(),
#             nt_stax.Dropout(rate=dropout_p),
#             nt_stax.Dense(out_dim=out_classes, W_std=W_std, b_std=b_std)
#          ]

#         self.conv_init_fn, self.conv_apply_fn = stax.serial(*conv_layers)
#         self.fc_init_fn, self.fc_apply_fn, self.fc_kernel_fn = nt_stax.serial(*fc_layers)

#         self.key = jax.random.PRNGKey(np.random.randint(1000000000) if seed is None else seed)
#         self.nn_key = None
#         self.conv_params = None
#         self.fc_params = None
#         self.conv_out_dim = None
#         self.init_weights()

#         self.use_empirical_kernel = use_empirical_kernel
#         self.fc_emp_kernel = jax.jit(
#             fun=nt.empirical_kernel_fn(partial(self.fc_apply_fn, rng=self.nn_key),
#                                        implementation=1),
#             static_argnames=['get']
#         )
#         self.fc_emp_kernel_diagonal = jax.jit(
#             fun=nt.empirical_kernel_fn(partial(self.fc_apply_fn, rng=self.nn_key),
#                                        diagonal_axes=(0,), implementation=1),
#             static_argnames=['get']
#         )

#     def init_weights(self):
#         self.key, conv_net_key = jax.random.split(self.key)
#         _, self.conv_params = self.conv_init_fn(conv_net_key, (-1, *self.in_dim))

#         self.key, self.nn_key = jax.random.split(self.key)
#         self.conv_out_dim = self.conv_apply_fn(self.conv_params, jnp.ones(shape=(1, *self.in_dim)),
#                                                rng=self.nn_key).shape[1]

#         self.key, fc_net_key = jax.random.split(self.key)
#         _, self.fc_params = self.fc_init_fn(conv_net_key, (-1, self.conv_out_dim))

#     def apply_conv(self, xs):
#         self.nn_key, key = jax.random.split(self.nn_key)
#         return self.conv_apply_fn(self.conv_params, xs, rng=key)

#     def apply_fc(self, xs):
#         self.nn_key, key = jax.random.split(self.nn_key)
#         return self.fc_apply_fn(self.fc_params, xs, rng=key)

#     def __call__(self, xs):
#         return self.apply_fc(self.apply_conv(xs))

#     def _get_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False, kernel: str = 'ntk'):
#         assert (not get_diagonal_only) or (x2 is None)
#         h1 = self.apply_conv(x1)
#         h2 = h1 if x2 is None else self.apply_conv(x2)
#         if self.use_empirical_kernel:
#             n1 = x1.shape[0]
#             n2 = n1 if x2 is None else x2.shape[0]
#             if get_diagonal_only:
#                 return self.fc_emp_kernel(x1=h1, x2=None, get=kernel, params=self.fc_params)
#             else:
#                 return self.fc_emp_kernel(x1=h1, x2=h2, get=kernel, params=self.fc_params).reshape(n1, n2)
#         else:
#             if get_diagonal_only:
#                 return jnp.diag(self.fc_kernel_fn(x1_or_kernel=h1, get=kernel))
#             else:
#                 return self.fc_kernel_fn(x1_or_kernel=h1, x2=h2, get=kernel)

#     def get_ntk(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
#         return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='ntk')

#     def get_nngp(self, x1: jnp.ndarray, x2: jnp.ndarray = None, get_diagonal_only: bool = False):
#         return self._get_kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only, kernel='nngp')