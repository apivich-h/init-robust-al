from typing import Optional, Callable, Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from al_ntk.dataset.base_dataset import Dataset as ALDataset


# simple dataset object to use with DataLoader 
class LoaderDataset(Dataset):

    def __init__(self, idxs, xs, ys, categorical_data: bool, device: torch.device = None):
        assert idxs.shape[0] == xs.shape[0] == ys.shape[0], (idxs.shape[0], xs.shape[0], ys.shape[0])
        self.device = device
        self.idxs = idxs
        
        self.xs = torch.tensor(xs, dtype=torch.get_default_dtype(), device=self.device)
        self.ys = torch.tensor(ys, dtype=torch.get_default_dtype(), device=self.device)
        if categorical_data:
            self.ys = self.ys.squeeze(-1).to(torch.long)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.xs[idx], self.ys[idx]


def collate(data):
    idxs, xs, ys = zip(*data)
    return idxs, torch.vstack(xs), torch.tensor(ys)


# return train and test data loaders
def data_loaders(dataset: ALDataset, selected: Iterable = None, 
                 batch_size: int = 32, test_batch_size: int = None,
                 shuffle: bool = True, device: torch.device = None,
                 collate_fn: Optional[Callable] = None, device_kwargs: Dict = dict(), 
                 build_test_loader: bool = True, generate_reg_data: bool = False, 
                 in_class_val: float = 1., not_in_class_val: float = -1.):
    
    if test_batch_size is None:
        test_batch_size = batch_size

    def loader(idxs, xs, ys, problem_type, do_shuffle, _batch_sz):
        dataset = LoaderDataset(idxs, xs, ys,  
                                categorical_data=(problem_type == 'classification' and
                                                  not generate_reg_data),
                                device=device)
        return DataLoader(
            dataset=dataset,
            batch_size=_batch_sz,
            shuffle=do_shuffle,
            collate_fn=collate_fn,
            **device_kwargs
        )
        
    if generate_reg_data:
        ys_train, ys_test = dataset.generate_label_for_regression(in_class_val=in_class_val,
                                                                  not_in_class_val=not_in_class_val)
    else:
        ys_train = dataset.ys_train
        ys_test = dataset.ys_test

    if selected is None:
        train_loader = loader(np.arange(dataset.train_count()), dataset.xs_train, ys_train,
                              problem_type=dataset.problem_type, do_shuffle=shuffle, _batch_sz=batch_size)
    else:
        assert np.unique(selected.flatten()).shape[0] == selected.shape[0]
        train_loader = loader(selected.flatten(), dataset.xs_train[selected], ys_train[selected],
                              problem_type=dataset.problem_type, do_shuffle=shuffle, _batch_sz=batch_size)

    if build_test_loader and (dataset.xs_test is not None):
        test_loader = loader(np.arange(0, dataset.xs_test.shape[0]), dataset.xs_test, ys_test,
                             dataset.problem_type, do_shuffle=False, _batch_sz=test_batch_size)
    else:
        test_loader = None

    return train_loader, test_loader


def get_torch_dataset(dataset: ALDataset, which_set: str = 'train', train_selected: np.ndarray = None,
                      return_indexs: bool = False):
    if which_set == 'train':
        if return_indexs:
            if train_selected is None:
                train_data_torch = TensorDataset(torch.arange(start=0, end=dataset.train_count()),
                                                 torch.tensor(dataset.xs_train, dtype=torch.get_default_dtype()),
                                                 torch.tensor(dataset.ys_train.flatten()))
            else:
                train_data_torch = TensorDataset(torch.arange(start=0, end=dataset.train_count())[train_selected],
                                                 torch.tensor(dataset.xs_train[train_selected, :], dtype=torch.get_default_dtype()),
                                                 torch.tensor(dataset.ys_train.flatten()[train_selected]))
        else:
            if train_selected is None:
                train_data_torch = TensorDataset(torch.tensor(dataset.xs_train, dtype=torch.get_default_dtype()),
                                                 torch.tensor(dataset.ys_train.flatten()))
            else:
                train_data_torch = TensorDataset(torch.tensor(dataset.xs_train[train_selected, :], dtype=torch.get_default_dtype()),
                                                 torch.tensor(dataset.ys_train.flatten()[train_selected]))
        return train_data_torch
    elif which_set == 'test':
        assert dataset.xs_test is not None
        if return_indexs:
            return TensorDataset(torch.arange(start=0, end=dataset.test_count()),
                                 torch.tensor(dataset.xs_test, dtype=torch.get_default_dtype()),
                                 torch.tensor(dataset.ys_test.flatten()))
        else:
            return TensorDataset(torch.tensor(dataset.xs_test, dtype=torch.get_default_dtype()),
                                 torch.tensor(dataset.ys_test.flatten()))
    else:
        raise ValueError('which_set must either be "train" or "test".')
