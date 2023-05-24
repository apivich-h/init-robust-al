import numpy as np

from neural_tangents.predict import gp_inference, Gaussian

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel


def get_gp(model: NNModel, dataset: Dataset = None, xs_train: np.ndarray = None, ys_train: np.ndarray = None,
           selected: np.ndarray = None, obs_noise: float = 0.1):
    if dataset is not None:
        xs_train, ys_train = dataset.get_train_data()
    else:
        assert xs_train is not None
        if ys_train is None:
            ys_train = np.zeros(shape=(xs_train.shape[0], 1))

    if selected is not None:
        xs_train = xs_train[selected, :]
        ys_train = ys_train[selected, :]
    gp_inf = gp_inference(k_train_train=model.kernel_fn(xs_train), y_train=ys_train, diag_reg=obs_noise)
    return gp_inf


def get_gp_covariance(model: NNModel, xs, xs_given=None, gp_inf=None, mode='ntk'):
    if xs_given is None or xs_given.shape[0] == 0:
        if mode == 'ntkgp':
            return model.get_ntk(xs)
        elif mode in {'ntk', 'nngp'}:
            return model.get_init_kernel(xs)
        else:
            raise ValueError('Invalid mode')
    else:
        if gp_inf is None:
            gp_inf = get_gp(model=model, xs_train=xs_given, obs_noise=1e-5)
        posterior = gp_inf(mode,
                           k_test_train=model.kernel_fn(xs, xs_given),
                           k_test_test=model.kernel_fn(xs, xs))
        return posterior.covariance


def predict_with_gp(model: NNModel, dataset: Dataset, xs_test: np.ndarray = None, selected: np.ndarray = None,
                    obs_noise: float = 0.1, mode: str = 'ntk'):
    gp_inf = get_gp(model=model, dataset=dataset, selected=selected, obs_noise=obs_noise)
    if selected is None:
        xs_train = dataset.xs_train
    else:
        xs_train = dataset.xs_train[selected, :]
    if xs_test is None:
        xs_test = dataset.xs_test
    posterior = gp_inf(mode,
                       k_test_train=model.kernel_fn(xs_test, xs_train),
                       k_test_test=model.kernel_fn(xs_test))
    return posterior


def get_model_sample_from_distribution(model: NNModel, dataset: Dataset, xs_test: np.ndarray = None,
                                       selected: np.ndarray = None, sample_count: int = 1, obs_noise: float = 0.1,
                                       mode: str = 'ntk'):
    posterior = predict_with_gp(model=model, dataset=dataset, xs_test=xs_test, selected=selected,
                                obs_noise=obs_noise, mode=mode)
    samples = np.empty(shape=(sample_count, posterior.mean.shape[0]))
    for i in range(sample_count):
        samples[i] = np.random.multivariate_normal(mean=posterior.mean.flatten(), cov=posterior.covariance)
    return samples
