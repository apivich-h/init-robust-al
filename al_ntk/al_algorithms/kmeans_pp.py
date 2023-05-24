import numpy as np
from sklearn.cluster._kmeans import kmeans_plusplus

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel
from .base_al_algorithm import ALAlgorithm


class KMeansPlusPlusMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel, max_proportion: float = 0.2):
        super().__init__(dataset, model)
        self.max_count = int(max_proportion * dataset.train_count())
        self.curr_count = 0
        xs = np.array(self.dataset.xs_train.reshape((self.dataset.xs_train.shape[0], -1)))
        _, self.idxs = kmeans_plusplus(X=xs, n_clusters=self.max_count, n_local_trials=10)


    def get(self, n: int):
        assert self.curr_count + n <= self.max_count
        self.selected[self.idxs[self.curr_count:self.curr_count + n]] = True
        self.curr_count += n
