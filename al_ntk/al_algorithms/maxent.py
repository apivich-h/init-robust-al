import numpy as np
import tqdm

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel
from .base_al_algorithm import ALAlgorithm
from al_ntk.utils.linalg import multidot, inverse_with_check


class MaxEntropyMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel, reg: float = 1e-4):
        super().__init__(dataset)
        self.model = model
        self.reg = reg
        self.K = np.array(self.model.get_ntk(self.dataset.xs_train))

    def get(self, n: int):

        m = np.sum(self.selected)
        K = self.K
        v_size = np.diag(K)

        for i in tqdm.trange(m, n + m):

            if i == 0:
                val = v_size

            else:
                A = K[np.ix_(self.selected, self.selected)]
                b = K[self.selected]
                val = v_size - multidot(b, inverse_with_check(A + self.reg * np.eye(A.shape[0])) @ b)

            val = np.ma.masked_array(val, mask=self.selected)
            best = val.argmax()
            self.selected[best] = True
            # self._debug[f'val_round_{i}'] = (val, best)