from copy import deepcopy

# import pdb
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances

import torch
from torch.nn import Sigmoid, Softmax

from al_ntk.al_algorithms.base_al_algorithm import ALAlgorithm
from al_ntk.dataset import Dataset
from al_ntk.model.nn_model import TorchNNModel
from al_ntk.utils.torch_dataloader import data_loaders, collate
from al_ntk.utils.maps import loss_map, optim_map


def prob_to_label(y, problem_type):

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if problem_type == 'binary_classification':
        return (y > 0).astype(int)
    elif problem_type == 'classification':
        return np.argmax(y, axis=-1)
    else:
        raise ValueError(f"problem type not recognized, received {problem_type}, expected one of 'binary classification' or 'classification'")


def softmax(out):

    if out.shape[-1] == 1:
        probs = Sigmoid()(out)
        return torch.cat([probs, 1-probs], axis=-1)
    else:
        return Softmax(dim=-1)(out)

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        # if sum(D2) == 0.0:
        #     pdb.set_trace()
        D2 = D2.ravel().astype(float)
        D2_copy = D2.copy()
        for i in indsAll:
            D2_copy[i] = 0.
        Ddist = (D2_copy ** 2) / sum(D2_copy ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        # while ind in indsAll:
        #     ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class BADGEMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: TorchNNModel, train: dict = None, optim: dict = None,
                 min_to_train: int = 0, reset_training: bool = True, use_cuda: bool = True,
                 break_acc: float = 0.99, max_attempts: int = 50):
        
        '''
        Parameters
        ----------
        dataset: dataset
        model: a TorchNNModel instance
        train: training configurations {batch_size, shuffle, epochs}
        optim: optimizer configurations {name, params}
        min_to_train: minimum samples needed before training to use badge
        reset_training: reinitialize fn weights every time you train
        '''
        
        assert dataset.problem_type in ('binary_classification', 'classification')
        assert isinstance(train, dict) 
        assert isinstance(optim, dict) 
        assert isinstance(min_to_train, int) 
        assert isinstance(reset_training, bool) 
        assert isinstance(use_cuda, bool)
        
        super().__init__(dataset, model)
        self.train_conf, self.optim_conf, self.min_to_train, self.reset = train, optim, min_to_train, reset_training
        self.verbose = True

        # training device configurations
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.model.to(self.device)
        self.device_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else dict()

        # loss class - inferred from problem type as in data
        Loss = loss_map[self.dataset.problem_type]
        self.criterion = Loss(reduction='none')
        self.break_acc = break_acc
        self.max_attempts = max_attempts

        self.last_batch_idxs = None

    def get(self, n: int):

        if self.get_selected().shape[0] < self.min_to_train:
            self.baseline_select(n)
        else:
            if self.selected.any() and self.train_conf is not None and self.optim_conf is not None:
                print('Do training for gradient embedding.')
                self.train()
            elif self.reset:
                print('Reset random init weights.')
                self.model.init_weights()
            idxs_unlabeled = np.arange(self.dataset.train_count())[~self.selected]
            grad_embeddings = self._get_grad_embedding()
            grad_embeddings = grad_embeddings[idxs_unlabeled]
            centers = init_centers(grad_embeddings, n)
            self.last_batch_idxs = idxs_unlabeled[centers]
            self.selected[self.last_batch_idxs] = True

    def baseline_select(self, n):

        idxs_unlabelled = np.arange(self.dataset.train_count())[~self.selected]
        self.last_batch_idxs = np.random.choice(idxs_unlabelled, n, replace=False)
        self.selected[self.last_batch_idxs] = True

    def train(self):

        if self.reset:
            self.model.init_weights()
        
        data_loader, _ = data_loaders(
            dataset=self.dataset, 
            selected=np.argwhere(self.selected).flatten(),
            device=self.device,
            **self.train_conf['data_loader_args']
        )
        # optimizer class
        optimizer = optim_map[self.optim_conf['name']](self.model.parameters(), **self.optim_conf['params'])

        self.model.train()
        
        epoch = epoch_acc = best_acc = attempts = 0 
        while epoch_acc < self.break_acc and attempts < self.max_attempts:
            epoch += 1
            epoch_acc = 0
            for _, xs, true_labels in data_loader:
                optimizer.zero_grad()
                pred_probs = self.model(xs)
                epoch_acc += np.sum(prob_to_label(pred_probs, self.dataset.problem_type) == true_labels.detach().cpu().numpy())
                loss = self.criterion(pred_probs, true_labels).mean()
                loss.backward()
                for p in filter(lambda p: p.grad is not None, self.model.parameters()):
                    p.grad.data.clamp_(min=-.1, max=.1)
                optimizer.step()
            epoch_acc /= len(data_loader.dataset)
            if best_acc < epoch_acc:
                best_acc, attempts = epoch_acc, 0
            else:
                attempts += 1
            if self.verbose:
                print(f'epoch: {epoch}, attempts: {attempts}, epoch accuracy: {epoch_acc}')
            if epoch >= self.train_conf['epochs'] and epoch_acc < 0.2:
                self.model.init_weights()
                optimizer = optim_map[self.optim_conf['name']](self.model.parameters(), **self.optim_conf['params'])

    # gradient embedding for badge (assumes cross-entropy loss)
    def _get_grad_embedding(self):

        data_loader, _ = data_loaders(
            dataset=self.dataset, 
            batch_size=256 if self.train_conf is None else self.train_conf['data_loader_args']['batch_size'],
            shuffle=False,
            device=self.device
        )
        n_labels = self.dataset.out_dim + (1 if self.dataset.problem_type == 'binary_classification' else 0)

        self.model.eval()
        with torch.no_grad():
            embedding = None
            for idxs, xs, _ in data_loader:
                p_out, out = self.model.get_penultimate_and_final_output(xs.to(self.device))
                p_out = p_out.data.cpu().numpy()
                pred_probs = softmax(out).data.cpu().numpy()
                if embedding is None:
                    emb_dim = p_out.shape[-1]
                    embedding = np.zeros([self.dataset.train_count(), emb_dim*n_labels])
                pred_classes = prob_to_label(pred_probs, self.dataset.problem_type)
                for i in range(xs.shape[0]):
                    for c in range(n_labels):
                        if c == pred_classes[i]:
                            embedding[idxs[i], emb_dim*c:emb_dim*(c+1)] = deepcopy(p_out[i]) * (1-pred_probs[i, c])
                        else:
                            embedding[idxs[i], emb_dim*c:emb_dim*(c+1)] = deepcopy(p_out[i]) * (-pred_probs[i, c])

        return embedding
