import numpy as np

from al_ntk.model import NNModel
from al_ntk.dataset import Dataset


class JudgeMetric:

    @staticmethod
    def get_value(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                  y_mean: float = 0., y_std: float = 1.0, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_results_string(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                           y_mean: float = 0., y_std: float = 1.0, **kwargs) -> str:
        raise NotImplementedError
