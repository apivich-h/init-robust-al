import numpy as np
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
from typing import Union, Callable

from .base_dataset import Dataset
from al_ntk.model.nn_model import NNModel, TorchNNModel, JaxNNModel


class MockFromModel(Dataset):

    def __init__(self, fn: Union[NNModel, Callable], n: int = 100, 
                 ball_scale: float = 5., seed: int = 42, noise: float = 1e-1,
                 distribution: str = 'uniform', problem_type: str = 'regression', shift_mean_y=True):
        self.ball_scale = ball_scale
        dim = fn.in_dim
        rand = np.random.RandomState(seed=seed)
        if distribution == 'uniform':
            xs_pool = rand.uniform(low=-ball_scale, high=ball_scale, size=(n, *dim))
            xs_test = rand.uniform(low=-ball_scale, high=ball_scale, size=(n, *dim))
        elif distribution == 'normal':
            xs_pool = (ball_scale / 3.) * rand.normal(size=(n, *dim))
            xs_test = (ball_scale / 3.) * rand.normal(size=(n, *dim))
        elif distribution == 'uniform_sphere':
            xs_pool = rand.normal(size=(n, *dim))
            xs_pool = ball_scale * (xs_pool.T / np.linalg.norm(xs_pool, axis=1)).T
            xs_test = rand.normal(size=(n, *dim))
            xs_test = ball_scale * (xs_test.T / np.linalg.norm(xs_test, axis=1)).T
        elif distribution == 'uniform_ball':
            xs_pool = rand.normal(size=(n, *dim))
            xs_pool = ball_scale * (xs_pool.T / (np.linalg.norm(xs_pool, axis=1) / np.sqrt(np.random.rand(n)))).T
            xs_test = rand.normal(size=(n, *dim))
            xs_test = ball_scale * (xs_test.T / (np.linalg.norm(xs_test, axis=1) / np.sqrt(np.random.rand(n)))).T
        else:
            raise ValueError('Invalid distribution: ' + distribution)
        
        if isinstance(fn, TorchNNModel):
            fn_torch = lambda xs: fn(
                torch.tensor(xs, dtype=torch.get_default_dtype(), device=fn.device)).cpu().detach().numpy()
            ys_pool = fn_torch(xs_pool)
            ys_test = fn_torch(xs_test)
        elif isinstance(fn, JaxNNModel):
            ys_pool = np.array(fn(jnp.array(xs_pool)))
            ys_test = np.array(fn(jnp.array(xs_test)))
        elif callable(fn):
            ys_pool = np.array(fn(xs_pool))
            ys_test = np.array(fn(xs_test))
        else:
            raise ValueError('fn is invalid.')
        
        out_dim = ys_pool.shape[1]
        
        ys_pool += noise * rand.randn(*ys_pool.shape)
        ys_test += noise * rand.randn(*ys_test.shape)

        if problem_type != 'classification' and shift_mean_y:
            # shift the mean to make things more balanced
            mean = np.mean(ys_pool)
            ys_pool = ys_pool - mean
            ys_test = ys_test - mean

        if problem_type == 'classification':
            ys_pool = np.argmax(ys_pool, axis=1).reshape(-1, 1)
            ys_test = np.argmax(ys_test, axis=1).reshape(-1, 1)
        elif problem_type == 'binary_classification':
            ys_pool = (ys_pool > 0.).astype(int)
            ys_test = (ys_test > 0.).astype(int)

        super().__init__(xs_pool, ys_pool, xs_test, ys_test, problem_type=problem_type, out_dim=out_dim)

    # def plot(self, plot_points=True, plot_orig_fn=True, selected=None, fn=None, error_fn=None):
    #     dim = self.input_dimensions()
    #
    #     if dim == 1:
    #         if plot_points:
    #             if selected is None:
    #                 plt.scatter(self.xs_train.flatten(), self.ys_train.flatten(), marker='x', color='black')
    #             else:
    #                 plt.scatter(self.xs_train[~selected], self.ys_train[~selected], marker='x', color='black', alpha=0.3)
    #                 plt.scatter(self.xs_train[selected], self.ys_train[selected], marker='o', color='blue')
    #
    #         xs_flat = np.linspace(-1.2 * self.ball_scale, 1.2 * self.ball_scale, num=101, endpoint=True)
    #         xs = xs_flat.reshape(-1, 1)
    #         if fn is not None:
    #             fn_val = fn(xs).flatten()
    #         else:
    #             fn_val = self.fn(xs).flatten()
    #         if plot_orig_fn or (fn is not None):
    #             plt.plot(xs_flat, fn_val, color='C0', alpha=0.7)
    #             if error_fn is not None:
    #                 error_val = error_fn(xs).flatten()
    #                 plt.fill_between(xs_flat, fn_val - error_val, fn_val + error_val, color='C0', alpha=0.2)
    #
    #     elif dim == 2:
    #         if plot_points:
    #             if selected is None:
    #                 plt.scatter(self.xs_train[:, 0], self.xs_train[:, 1], marker='x', color='black')
    #             else:
    #                 plt.scatter(self.xs_train[~selected, 0], self.xs_train[~selected, 1], marker='x', color='black',
    #                             alpha=0.3)
    #                 plt.scatter(self.xs_train[selected, 0], self.xs_train[selected, 1], marker='o', color='C1')
    #
    #         if plot_orig_fn or (fn is not None):
    #             xs = np.linspace(-1.2 * self.ball_scale, 1.2 * self.ball_scale, num=51, endpoint=True).reshape(-1, 1)
    #             xs_mesh_plot = np.meshgrid(xs, xs)
    #             xs_test_al = np.stack([xx.flatten() for xx in xs_mesh_plot]).T
    #             if fn is not None:
    #                 ys_test_al = fn(xs_test_al)
    #             else:
    #                 ys_test_al = self.fn(xs_test_al)
    #             mesh_val = ys_test_al.reshape(51, 51)
    #             cs = plt.contour(*xs_mesh_plot, mesh_val, alpha=0.7)
    #             plt.clabel(cs, inline=True, fontsize=10)
    #
    #     else:
    #         raise ValueError('Dimension must be 1 or 2 to be plottable.')
