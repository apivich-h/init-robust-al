import numpy as np

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel
from .base_al_algorithm import ALAlgorithm


class RandomMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel):
        super().__init__(dataset, model)
        self.rand_perm = np.random.permutation(dataset.train_count())
        self.last_selected = 0

    def get(self, n: int):
        for _ in range(n):
            added = False
            while not added:
                x = self.rand_perm[self.last_selected]
                if not self.selected[x]:
                    self.selected[x] = True
                    self.last_selected += 1
                    added = True
