import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from .base_dataset import Dataset


class MockUnbalancedCluster(Dataset):

    def __init__(self, fn=None, dim=2, n=100, n_clusters=2, seed=42, evenly_distribute_tests=True,
                 problem_type='regression', shift_mean_y=True):
        self.dim = dim
        rand = np.random.RandomState(seed)
        X_clust, _ = make_blobs(n_samples=2*n,
                                n_features=dim,
                                centers=np.vstack([np.zeros((n_clusters, dim)),
                                                   2. * np.sign(rand.randn(n_clusters, dim)) / np.sqrt(dim)]),
                                cluster_std=np.concatenate([np.ones(n_clusters), 0.3 * np.ones(n_clusters)]),
                                random_state=rand)

        xs_pool = X_clust[:n]
        if evenly_distribute_tests:
            xs_test = 1. * rand.normal(size=(n, dim))
        else:
            xs_test = X_clust[n:]

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
            classification_classes = ys_pool.shape[1]
            ys_pool = np.argmax(ys_pool, axis=1).reshape(-1, 1)
            ys_test = np.argmax(ys_test, axis=1).reshape(-1, 1)
        elif problem_type == 'binary_classification':
            classification_classes = None
            ys_pool = (ys_pool > 0.).astype(int)
            ys_test = (ys_test > 0.).astype(int)
        else:
            classification_classes = None

        super().__init__(xs_pool, ys_pool, xs_test, ys_test,
                         problem_type=problem_type, out_dim=classification_classes)
