import numpy as np
from scipy.stats import norm

from .base_metric import JudgeMetric
from al_ntk.dataset import Dataset
from al_ntk.model import NNModel
from al_ntk.utils.nn_predict import predict_with_gp


class NLLH(JudgeMetric):

    @staticmethod
    def get_value(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                  y_mean: float = 0., y_std: float = 1.0, **kwargs):
        posterior = predict_with_gp(model=model, dataset=dataset, selected=selected, **kwargs)
        y_pred_mean = posterior.mean
        y_pred_var = np.diag(posterior.covariance)
        y_pred_mean = (y_std * y_pred_mean.reshape(dataset.ys_test.shape)) + y_mean
        y_pred_std = y_std * np.sqrt(y_pred_var.reshape(dataset.ys_test.shape))
        log_llh = -norm.logpdf(x=(y_std * dataset.ys_test) + y_mean, loc=y_pred_mean, scale=y_pred_std)
        return np.mean(log_llh)

    @staticmethod
    def get_results_string(model: NNModel, dataset: Dataset, selected: np.ndarray = None,
                           y_mean: float = 0., y_std: float = 1.0, **kwargs) -> str:
        nllh = NLLH.get_value(model=model, dataset=dataset, selected=selected, y_mean=y_mean, y_std=y_std, **kwargs)
        return f'nllh: {nllh:.10f}'
