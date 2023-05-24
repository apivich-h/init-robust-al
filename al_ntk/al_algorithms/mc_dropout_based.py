from dataclasses import dataclass
from typing import List, Callable, Iterable
import math
from copy import deepcopy

import numpy as np
import torch
from toma import toma
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as tdata
from tqdm.auto import tqdm

from baal import ActiveLearningDataset, ModelWrapper
from baal.active import ActiveLearningLoop
from baal.active.heuristics import BatchBALD, BALD, Entropy, Certainty, Variance, Margin
from baal.bayesian.dropout import patch_module

from al_ntk.dataset.base_dataset import Dataset
from al_ntk.model.nn_model import TorchNNModel
from al_ntk.al_algorithms.base_al_algorithm import ALAlgorithm
from al_ntk.utils.torch_dataloader import data_loaders, get_torch_dataset
from al_ntk.utils.maps import loss_map, optim_map

_scoring_metric = {
    'batchbald': BatchBALD,
    'bald': BALD,
    'entropy': Entropy,
    'certainty': Certainty,
    'variance': Variance,
    'margin': Margin
}


class MCDropoutBasedMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: TorchNNModel, train, optim,
                 heuristic: str = 'batchbald', use_cuda: bool = True, first_batch_random: bool = False,
                 num_inference_samples: int = 20, num_test_inference_samples: int = 5, **kwargs):
        """ Wrapper class for active learning with BAAL

        Parameters
        ----------
        dataset: dataset to use
        model: fn to use (should be extended from TorchNNModel, and should involve MCDropout)
        query_size: how many queries to be obtained in each loop
        heuristic: which scoring metric should be used for AL
        device: device for CUDA
        opt_generator: function that returns optimiser that should be used for training
            It should input the fn parameter (i.e. fn.parameter)
            If None, use the default SGD optimiser
        """
        super().__init__(dataset=dataset, model=model)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # params for the AL process
        self.num_inference_samples = num_inference_samples
        self.num_test_inference_samples = num_test_inference_samples
        self.num_samples = self.dataset.train_count()
        self.batch_size = train['data_loader_args']['batch_size']
        self.epochs = train['epochs']
        
        assert heuristic in _scoring_metric.keys()
        self.heuristic_name = heuristic
        self.heuristic_kwargs = kwargs
        self.first_batch_random = first_batch_random

        # init new fn for the round
        self.model = self.model.to(device=self.device)
        self.model = patch_module(self.model)
        self.initial_weights = deepcopy(self.model.state_dict())

        if self.use_cuda:
            self.model = self.model.cuda()
        self.wrapper = ModelWrapper(model=self.model, criterion=nn.CrossEntropyLoss())

        # if opt_generator is None:
        #     self.optim_generator = lambda params: optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
        # else:
        #     self.optim_generator = opt_generator

        self.train_dset = get_torch_dataset(self.dataset, which_set='train')
        self.test_dset = get_torch_dataset(self.dataset, which_set='test')
        self.al_dataset = ActiveLearningDataset(self.train_dset)

        self.optimiser_generator = lambda params: optim_map[optim['name']](params, **optim['params'])

        # Setup our active learning loop for our experiments
        self.al_loop = None
        self.query_size = None
        self.flag = True

    def preselect(self, idxs):
        assert not self.selected.any()
        self.selected[idxs] = True
        if self.selected.any():
            self.al_dataset.label(list(idxs))

    def get(self, n: int):

        if not self.selected.any():
            if self.first_batch_random:
                self.preselect(np.random.permutation(self.dataset.train_count())[:n])
                return
            else:
                self.preselect(np.arange(0))

        if self.al_loop is None:
            self.query_size = n
            
            if self.heuristic_name == 'batchbald' and 'num_samples' not in self.heuristic_kwargs.keys():
                self.heuristic_kwargs['num_samples'] = self.query_size
            self.heuristic = _scoring_metric[self.heuristic_name](**self.heuristic_kwargs)
            
            self.al_loop = ActiveLearningLoop(
                dataset=self.al_dataset,
                get_probabilities=self.wrapper.predict_on_dataset,
                heuristic=self.heuristic,
                query_size=self.query_size,
                iterations=self.num_inference_samples,
                batch_size=self.batch_size,
                use_cuda=self.use_cuda,
                verbose=False,
                workers=2,  # can this be fixed
            )
        else:
            assert n == self.query_size

        assert self.flag, "Exceeded budget"
        self.model.load_state_dict(self.initial_weights)
        if self.selected.any():
            train_loss = self.wrapper.train_on_dataset(
                self.al_dataset,
                optimizer=self.optimiser_generator(self.model.parameters()),
                batch_size=self.batch_size,
                epoch=self.epochs,
                use_cuda=self.use_cuda,
                workers=2  # can this be fixed
            )

        # print(self.wrapper.get_metrics())
        self.flag = self.al_loop.step(self.al_dataset.pool)

        self.selected = self.al_dataset.labelled
