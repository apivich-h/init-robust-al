import numpy as np


class Dataset:

    def __init__(self, xs_train, ys_train, xs_test=None, ys_test=None,
                 problem_type: str = 'regression', out_dim: int = None):
                 
        self._check_dims(xs_train, ys_train, xs_test, ys_test, problem_type=problem_type)
        self.xs_train = np.array(xs_train, dtype=np.float64)
        self.ys_train = np.array(ys_train, dtype=np.float64).reshape(xs_train.shape[0], -1)

        if xs_test is None:
            self.xs_test = None
            self.ys_test = None
        else:
            self.xs_test = np.array(xs_test, dtype=np.float64)
            self.ys_test = np.array(ys_test, dtype=np.float64).reshape(xs_test.shape[0], -1)

        if problem_type == 'classification':
            self.ys_train = self.ys_train.astype(np.long)
            self.out_dim = (out_dim if out_dim is not None
                            else np.max(self.ys_train) + 1)
            if self.ys_test is not None:
                self.ys_test = self.ys_test.astype(np.long)
                self.out_dim = (out_dim if out_dim is not None
                                else max(self.out_dim, np.max(self.ys_train) + 1))
        elif problem_type == 'binary_classification':
            self.ys_train = self.ys_train.astype(np.long)
            self.out_dim = 1
            if self.ys_test is not None:
                self.ys_test = self.ys_test.astype(np.long)
        else:
            self.out_dim = self.ys_train.shape[1]

        self.problem_type = problem_type
        self.dim = xs_train.shape[1]

    def reload_data(self, xs_train=None, ys_train=None, xs_test=None, ys_test=None):
        if xs_train is not None:
            self.xs_train = np.array(xs_train)
            self.ys_train = np.array(ys_train)
        if xs_test is not None:
            self.xs_test = np.array(xs_test)
            self.ys_test = np.array(ys_test)

    def _check_dims(self, xs_train, ys_train, xs_test, ys_test, problem_type):
        assert problem_type in {'regression', 'binary_classification', 'classification'}
        assert xs_train.shape[0] == ys_train.shape[0]
        if xs_test is not None:
            assert ys_test is not None
            assert xs_train.shape[1:] == xs_test.shape[1:]
            assert xs_test.shape[0] == ys_test.shape[0]

    def get_train_data(self):
        return self.xs_train, self.ys_train

    def get_test_data(self):
        return self.xs_test, self.ys_test

    def train_count(self):
        return self.xs_train.shape[0]

    def test_count(self):
        return self.xs_test.shape[0]

    def input_dimensions(self):
        return self.xs_train.shape[1:]

    def output_dimensions(self):
        if self.problem_type == 'regression':
            return self.ys_train.shape[1]
        elif self.problem_type == 'binary_classification':
            return 1
        else:
            return self.out_dim

    def generate_label_for_regression(self, in_class_val=1., not_in_class_val=-1., train_mask=None, test_mask=None):
        """ Generate the data which can be used for training with MSE

        Parameters
        ----------
        not_in_class_val: default value for when x does not belong to some class (only for multiclass classification)
            i.e. for traditional one-hot vectors, this value would be 0
        train_mask: indices of training elements which should be set to nan
        test_mask: indices of testing elements which should be set to nan

        Returns
        -------

        """
        if self.problem_type == 'regression':
            _ys_train = np.copy(self.ys_train)
            _ys_test = np.copy(self.ys_test)
        elif self.problem_type == 'classification':
            assert not_in_class_val < in_class_val
            embedding = not_in_class_val + (in_class_val - not_in_class_val) * np.eye(self.out_dim)
            _ys_train = embedding[self.ys_train.astype(np.int).flatten()]
            _ys_test = embedding[self.ys_test.astype(np.int).flatten()]
        elif self.problem_type == 'binary_classification':
            _ys_train = not_in_class_val + (in_class_val - not_in_class_val) * (self.ys_train > 0)
            _ys_test = not_in_class_val + (in_class_val - not_in_class_val) * (self.ys_test > 0)
        else:
            raise ValueError

        if train_mask is not None:
            _ys_train[train_mask, :] = np.nan
        if test_mask is not None:
            _ys_test[test_mask, :] = np.nan

        return _ys_train, _ys_test
