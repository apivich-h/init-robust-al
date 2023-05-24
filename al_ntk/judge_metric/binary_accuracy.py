# import numpy as np
# from scipy.stats import norm
#
# from .base_metric import JudgeMetric
# from al_ntk.dataset import Dataset
# from al_ntk.fn import NNModel
# from al_ntk.utils.nn_predict import predict_with_gp
#
#
# class BinaryAccuracy(JudgeMetric):
#
#     @staticmethod
#     def get_value(fn: NNModel, dataset: Dataset, selected: np.ndarray = None,
#                   y_mean: float = 0., y_std: float = 1.0, **kwargs):
#         assert dataset.ys_test.shape[1] == 1
#         posterior = predict_with_gp(fn=fn, dataset=dataset, selected=selected, **kwargs)
#         y_pred_var = (y_std ** 2) * np.diag(posterior.covariance)
#         ent = 0.5 * np.log(y_pred_var) + 0.5 * (1. + np.log(2. * np.pi))
#         return np.mean(ent)
#
#     @staticmethod
#     def get_results_string(fn: NNModel, dataset: Dataset, selected: np.ndarray = None,
#                            y_mean: float = 0., y_std: float = 1.0, **kwargs) -> str:
#         ent = Entropy.get_value(fn=fn, dataset=dataset, selected=selected,
#                                 y_mean=y_mean, y_std=y_std, **kwargs)
#         return f'entropy: {ent:.10f}'
