import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from .base_dataset import Dataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _create_EMNIST_dataset(split='mnist', num_repetitions: int = 1, subset_count: int = None, subset_method: str = 'random',
                           do_normalise: bool = True, add_noise: bool = True, seed: int = 42):

    transform_list = [transforms.ToTensor()]
    if add_noise:
        transform = transform_list.append(AddGaussianNoise(0., 0.1))
    # fixed for emnist datasets
    if do_normalise:
        transform = transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.EMNIST(root='~/emnist_data', split=split, train=True, download=True, transform=transform)
    
    if (subset_count is not None) and (subset_count < len(train_dataset)):
        
        rand = np.random.RandomState(seed=seed)
        
        if subset_method == 'random':
            indices = rand.choice(len(train_dataset), size=subset_count, replace=False)
            
        elif subset_method == 'power':
            probs = np.exp(-0.05 * np.arange(100))
            rand_no = rand.rand(len(train_dataset))
            indices = []
            for i, (_, y) in enumerate(train_dataset):
                if rand_no[i] < probs[y]:
                    indices.append(i)
            indices = rand.choice(indices, size=subset_count, replace=False)
    
        else:
            raise ValueError('Invalid subset_method.')
        
        train_dataset = data.Subset(train_dataset, indices=indices)

    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    test_dataset = datasets.EMNIST(root='~/emnist_data', split=split, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


class EMNIST(Dataset):
    
    def __init__(self, data_split='mnist', num_repetition: int = 1, subset_count: int = None, subset_method: str = 'random',
                 do_normalise: bool = True, add_noise: bool = False, flatten_input: bool = False, seed: int = 42):
        self.train_dataset, self.test_dataset = _create_EMNIST_dataset(
            split=data_split,
            num_repetitions=num_repetition,
            subset_count=subset_count,
            subset_method=subset_method,
            do_normalise=do_normalise,
            add_noise=add_noise,
            seed=seed
        )
        
        xs_train = np.empty(shape=(len(self.train_dataset), 1, 28, 28))
        ys_train = np.empty(shape=(len(self.train_dataset), 1))
        xs_test = np.empty(shape=(len(self.test_dataset), 1, 28, 28))
        ys_test = np.empty(shape=(len(self.test_dataset), 1))
        
        for i, (x, y) in enumerate(self.train_dataset):
            xs_train[i] = x.reshape((1, 28, 28))
            ys_train[i, 0] = y
            
        for i, (x, y) in enumerate(self.test_dataset):
            xs_test[i] = x.reshape((1, 28, 28))
            ys_test[i, 0] = y
            
        if flatten_input:
            xs_train = xs_train.reshape(len(self.train_dataset), -1)
            xs_test = xs_test.reshape(len(self.test_dataset), -1)
        
        super().__init__(xs_train=xs_train, ys_train=ys_train,
                         xs_test=xs_test, ys_test=ys_test,
                         problem_type='classification')
