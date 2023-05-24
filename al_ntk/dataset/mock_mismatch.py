import numpy as np
import matplotlib.pyplot as plt

from .base_dataset import Dataset


class MockMismatched(Dataset):

    def __init__(self, fn=None, dim=2, n=100, mean_dist=1., seed=42, problem_type='regression', shift_mean_y=True):

        self.dim = dim
        rand = np.random.RandomState(seed=seed)
        xs_pool = rand.normal(loc=mean_dist/np.sqrt(dim), scale=1., size=(n, dim))
        xs_test = rand.normal(loc=-mean_dist/np.sqrt(dim), scale=1., size=(n, dim))
        if fn is None:
            raise ValueError
        else:
            ys_pool = fn(xs_pool)
            ys_test = fn(xs_test)

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

        super().__init__(xs_pool, ys_pool, xs_test, ys_test, problem_type=problem_type)

    # def plot(self, show_test=False, selected=None, fn=None, levels=None):
    #     assert self.dim == 2, "Plotting only for 2D data"
    #
    #     xs = np.linspace(np.min([self.xs_train, self.xs_test]),
    #                      np.max([self.xs_train, self.xs_test]),
    #                      num=51, endpoint=True).reshape(-1, 1)
    #     xs_mesh = np.meshgrid(xs, xs)
    #     plt.figure(figsize=(7, 7))
    #     if isinstance(fn, np.ndarray):
    #         xs_mesh_plot = fn
    #     elif fn is not None:
    #         xs_mesh_plot = fn(np.stack([xx.flatten() for xx in xs_mesh]).T).reshape(51, 51)
    #
    #     if fn is not None:
    #         cs = plt.contour(*xs_mesh, xs_mesh_plot, levels=levels, alpha=0.7)
    #         plt.clabel(cs, inline=True, fontsize=10)
    #
    #     if selected is None:
    #         plt.scatter(self.xs_train[:, 0],
    #                     self.xs_train[:, 1],
    #                     marker='o', color='black')
    #         if show_test:
    #             plt.scatter(self.xs_test[:, 0],
    #                         self.xs_test[:, 1],
    #                         marker='x', color='red')
    #     else:
    #         plt.scatter(self.xs_train[~selected][:, 0],
    #                     self.xs_train[~selected][:, 1],
    #                     marker='o', color='black', alpha=0.3)
    #         plt.scatter(self.xs_train[selected, 0], self.xs_train[selected, 1], marker='o', color='blue')
    #         if show_test:
    #             plt.scatter(self.xs_test[:, 0],
    #                         self.xs_test[:, 1],
    #                         marker='x', color='red', alpha=0.5)
