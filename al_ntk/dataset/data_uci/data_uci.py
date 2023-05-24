import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from typing import Union

from ..base_dataset import Dataset

"""
This section of the code is adapted from
https://github.com/ratschlab/bnn_priors/tree/main/bnn_priors.

The original code was written for reading data_covertype into a torch-friendly format,
and put into a custom class. 
We use the compiled data_covertype, however then keep the read data_covertype as numpy format.
"""

UCI_SETS_REGRESSION = [
    'boston',          # 506 13
    'concrete',        # 1030 8
    'energy',          # 768 8
    'kin8nm',          # 8192 8
    'naval',           # 11934 16
    'optdigits_regr',  # 5620 64
    'power',           # 9568 4
    'protein',         # 45730 9
    'test2',           # 4000 16
    'test_sin',        # 10000 16
    'wine',            # 1599 11
    'yacht',           # 308 6
]

UCI_SETS_BINARY_CLASSIFICATION = [
    'breast_cancer_wisconsin',  # 683 9
    'spambase'                  # 4601 57
]

_UCI_SETS_CLASSIFICATION_CLASSES = {
    'image_segmentation': 7,  # 2310 19 7
    'optdigits': 10,          # 5620 64 10
    'test': 5,
    'yeast': 10,              # 1484 8 10
}

UCI_SETS_CLASSIFICATION = list(_UCI_SETS_CLASSIFICATION_CLASSES.keys())

UCI_SETS_ALL = UCI_SETS_REGRESSION + UCI_SETS_BINARY_CLASSIFICATION + UCI_SETS_CLASSIFICATION


def _process_obtained_data(train_X, train_y, test_X, test_y, is_classification, normalise_x, normalise_y):
    if is_classification:
        train_y = train_y.astype(np.int)
        test_y = test_y.astype(np.int)

    if normalise_x:
        # compute normalization constants based on training set
        X_std = np.std(train_X, axis=0)
        X_std[X_std == 0] = 1.  # ensure we don't divide by zero
        X_mean = np.mean(train_X, axis=0)
        train_X = (train_X - X_mean) / X_std
        test_X = (test_X - X_mean) / X_std
    else:
        X_std = 1.
        X_mean = 0.

    if normalise_y:
        y_mean = np.mean(train_y, axis=0)
        y_std = np.std(train_y, axis=0)
        train_y = (train_y - y_mean) / y_std
        test_y = (test_y - y_mean) / y_std
    elif is_classification:
        y_mean = 0
        y_std = 1
    else:
        y_mean = 0.
        y_std = 1.

    return train_X, train_y, test_X, test_y, X_mean, X_std, y_mean, y_std


def _load_uci_dataset(dataset, normalise_x: bool = True, normalise_y: bool = False,
                      whole_data_pool_prop: float = 1., split_method: str = 'random', split_prop: float = 0.1,
                      split_param: Union[np.ndarray, float] = None, random_state: int = 42):

    assert dataset in UCI_SETS_ALL
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = f'{_ROOT}/{dataset}/'
    data = np.loadtxt(f'{dataset_dir}/data.txt').astype(np.float32)
    index_features = np.loadtxt(f'{dataset_dir}/index_features.txt')
    index_target = np.loadtxt(f'{dataset_dir}/index_target.txt')
    X_unnorm = data[:, index_features.astype(int)]
    y_unnorm = data[:, index_target.astype(int):index_target.astype(int) + 1]
    X_unnorm, y_unnorm = shuffle(X_unnorm, y_unnorm, random_state=random_state-2)

    if whole_data_pool_prop < 1.:
        _, X_unnorm, _, y_unnorm = train_test_split(X_unnorm, y_unnorm,
                                                    test_size=whole_data_pool_prop,
                                                    random_state=random_state-1)

    if split_method == 'random':
        train_X, test_X, train_y, test_y = train_test_split(X_unnorm, y_unnorm,
                                                            test_size=split_prop,
                                                            random_state=random_state)

    elif split_method == 'unbalanced_labels':
        split_param = np.array(split_param)
        assert dataset in UCI_SETS_CLASSIFICATION or dataset in UCI_SETS_BINARY_CLASSIFICATION
        y_unnorm = y_unnorm.astype(np.int)
        thr = split_param[y_unnorm.flatten()]
        rand = np.random.RandomState(seed=random_state)
        in_test_idx = np.argwhere(rand.random(size=X_unnorm.shape[0]) < thr).flatten()
        if in_test_idx.shape[0] > split_prop * X_unnorm.shape[0]:
            rand.shuffle(in_test_idx)
            in_test_idx = in_test_idx[:int(split_prop * X_unnorm.shape[0])]
        in_test = np.zeros(shape=X_unnorm.shape[0], dtype=bool)
        in_test[in_test_idx] = True

        train_X = X_unnorm[~in_test, :]
        train_y = y_unnorm[~in_test, :]
        test_X = X_unnorm[in_test, :]
        test_y = y_unnorm[in_test, :]

    elif split_method == 'unbalanced_region':
        assert isinstance(split_param, np.ndarray)
        np.random.seed(seed=42)  # set seed
        n_cluster = split_param.shape[0]
        assert n_cluster > 0
        kmeans = KMeans(n_clusters=n_cluster).fit(X_unnorm)
        thr = split_param[kmeans.labels_.flatten()]
        rand = np.random.RandomState(seed=random_state)
        in_test_idx = np.argwhere(rand.random(size=X_unnorm.shape[0]) < thr).flatten()
        if in_test_idx.shape[0] > split_prop * X_unnorm.shape[0]:
            rand.shuffle(in_test_idx)
            in_test_idx = in_test_idx[:int(split_prop * X_unnorm.shape[0])]
        in_test = np.zeros(shape=X_unnorm.shape[0], dtype=bool)
        in_test[in_test_idx] = True

        train_X = X_unnorm[~in_test, :]
        train_y = y_unnorm[~in_test, :]
        test_X = X_unnorm[in_test, :]
        test_y = y_unnorm[in_test, :]

    else:
        raise ValueError('Invalid split_method.')

    if dataset in UCI_SETS_CLASSIFICATION:
        prob_type = 'classification'
        normalise_y = False  # for classification make sure not to normalise y
    elif dataset in UCI_SETS_BINARY_CLASSIFICATION:
        prob_type = 'binary_classification'
        normalise_y = False
    elif dataset in UCI_SETS_REGRESSION:
        prob_type = 'regression'
    else:
        raise ValueError("dataset not in available data")

    train_X, train_y, test_X, test_y, X_mean, X_std, y_mean, y_std = _process_obtained_data(
        train_X=train_X, train_y=train_y,
        test_X=test_X, test_y=test_y,
        normalise_x=normalise_x, normalise_y=normalise_y,
        is_classification=(prob_type == 'classification')
    )
    return train_X, train_y, test_X, test_y, X_mean, X_std, y_mean, y_std, prob_type


class UCI(Dataset):

    def __init__(self, dataset: str, normalise_x: bool = True, normalise_y: bool = True,
                 whole_data_pool_prop: float = 1., random_state: int = 42,
                 test_split_method: str = 'random', test_split_prop: float = 0.1,
                 test_split_param: Union[np.ndarray, float] = None):
        """ Constructor

        Parameters
        ----------
        dataset: name of UCI dataset to use
        normalise_x: whether the input should be normalised or not, i.e. transformed into (x - mean / std)
        normalise_y: whether the output should be normalised
        whole_data_pool_prop: proportion of the full dataset to use for train+test data points
        random_state: random seed
        test_split_method: how the train/test set should be split
        test_split_prop: proportion of the data pool to be set as test set
        test_split_param: extra parameters for the test set splitting method
        """
        train_X, train_y, test_X, test_y, X_mean, X_std, y_mean, y_std, prob_type = _load_uci_dataset(
            dataset=dataset,
            random_state=random_state,
            normalise_x=normalise_x,
            normalise_y=normalise_y,
            whole_data_pool_prop=whole_data_pool_prop,
            split_method=test_split_method,
            split_prop=test_split_prop,
            split_param=test_split_param,
        )
        super().__init__(
            train_X, train_y, test_X, test_y,
            problem_type=prob_type,
            out_dim=_UCI_SETS_CLASSIFICATION_CLASSES[dataset] if prob_type == 'classification' else None
        )
        self.x_mean = X_mean
        self.x_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std
