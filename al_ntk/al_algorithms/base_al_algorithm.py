from typing import Dict, Any, Union

import numpy as np

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel


class ALAlgorithm:

    def __init__(self, dataset: Dataset, model: NNModel):
        self.dataset = dataset
        self.model = model
        self.selected = np.zeros(self.dataset.train_count(), bool)
        self._debug = dict()

    def preselect(self, idxs: np.ndarray):
        """ Set some data as already queried before the active learning loop begins

        Parameters
        ----------
        idxs: indices which should be set as already selected

        Returns
        -------

        """
        assert not (self.selected[idxs]).any()
        self.selected[idxs] = True

    def get(self, n: int):
        """ Get a batch of elements

        Parameters
        ----------
        n: how many elements to select

        Returns
        -------
        """
        raise NotImplementedError

    def get_selected(self):
        return np.argwhere(self.selected).flatten()

    def get_debug(self):
        return self._debug
