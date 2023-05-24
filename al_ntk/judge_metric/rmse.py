import numpy as np

from .base_metric import JudgeMetric
from al_ntk.dataset import Dataset
from al_ntk.model import NNModel
from al_ntk.utils.nn_predict import get_model_sample_from_distribution


class RMSE(JudgeMetric):

    @staticmethod
    def get_value(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                  y_mean: float = 0., y_std: float = 1.0, sample_count: int = 100):
        y_pred = get_model_sample_from_distribution(model=model, dataset=dataset,
                                                    selected=selected, sample_count=sample_count)
        y_pred = y_pred.reshape((sample_count, dataset.ys_test.shape[0], dataset.ys_test.shape[1]))
        rmse_vals = np.empty(shape=(sample_count,))
        for i in range(sample_count):
            rmse_vals[i] = y_std * np.sqrt(np.mean((y_pred[i] - dataset.ys_test) ** 2))
        return np.mean(rmse_vals), np.std(rmse_vals), np.sort(rmse_vals)

    @staticmethod
    def get_results_string(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                           y_mean: float = 0., y_std: float = 1.0, **kwargs) -> str:
        rmse_mean, rmse_std, rmse_all = RMSE.get_value(model=model, dataset=dataset, selected=selected,
                                                       y_mean=y_mean, y_std=y_std, **kwargs)
        rmse_all_str = '[' + ', '.join(f'{x:.10f}' for x in rmse_all) + ']'
        return f'rmse_mean: {rmse_mean:.10f}\nrmse_std: {rmse_std:.10f}\nrmse_all: {rmse_all_str}'
