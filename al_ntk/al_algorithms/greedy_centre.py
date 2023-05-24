import numpy as np
import tqdm
from typing import Union, Callable, Dict, Any, Iterable
import torch
from torch.utils.data import DataLoader

from al_ntk.dataset import Dataset
from al_ntk.model import NNModel, JaxNNModel, TorchNNModel
from .base_al_algorithm import ALAlgorithm
from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.nn_training_jax import train_mlp
from al_ntk.utils.nn_training_torch import train_torch
from al_ntk.utils.maps import loss_map, optim_map


class GreedyCentreMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: Union[JaxNNModel, TorchNNModel] = None,
                 train: dict = None, optim: dict = None, use_cuda: float = True,
                 embedding_mode: str = 'output', do_ilp: bool = False, outlier_prop: float = 1e-3):
        """

        Parameters
        ----------
        dataset
        model
        embedding_mode: what to use as embedding
            if set to output, this matches method from Active-Coreset by Sener and Savarse
        do_ilp: perform ILP to get max cover or not (need cvxpy/gurobi to run though)
        outlier_prop: proportion of outlier to ignore in ILP
        train_model_function: If using output embedding, this function is used to train the fn in order to get
            the proper output embedding
        """
        super().__init__(dataset=dataset, model=model)
        self.embedding_mode = embedding_mode
        self.do_ilp = do_ilp
        self.outlier_prop = outlier_prop

        if train is not None:
            self.train_epochs = train['epochs']
            self.train_batch_sz = train['data_loader_args']['batch_size']
        else:
            self.train_epochs = None
            self.train_batch_sz = None
        
        if optim is not None:
            self.optimiser_generator = lambda params: optim_map[optim['name']](params, **optim['params'])
        else:
            self.optimiser_generator = None

        self.use_cuda = use_cuda and torch.cuda.is_available() and (self.model is not None)
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        if self.embedding_mode == 'ntk':
            assert isinstance(self.model, JaxNNModel)
            K = self.model.get_ntk(self.dataset.xs_train)
            cov = np.diag(K)
            cov_rep = np.repeat(cov[:, np.newaxis], cov.shape[0], 1)
            self.dist = np.sqrt(cov_rep + cov_rep.T - 2 * K)
            self._debug['dist_mat'] = self.dist
        elif self.embedding_mode == 'input':
            xs = self.dataset.xs_train
            xs = xs.reshape(xs.shape[0], -1)
            K = xs @ xs.T
            cov = np.diag(K)
            cov_rep = np.repeat(cov[:, np.newaxis], cov.shape[0], 1)
            self.dist = np.sqrt(cov_rep + cov_rep.T - 2 * K)
            self._debug['dist_mat'] = self.dist
        elif self.embedding_mode == 'output':
            self.dist = None
        else:
            raise ValueError('Invalid embedding_mode')

        self.last_batch_idxs = None

    def _solve_ilp(self, n: int, radius: float):
        import cvxpy as cp

        train_n = self.dataset.train_count()
        is_far = self.dist > radius

        selected = cp.Variable(shape=(train_n,), boolean=True)
        covered = cp.Variable(shape=(train_n, train_n), boolean=True)
        outlier = cp.Variable(shape=(train_n, train_n), boolean=True)

        constraints = [
            cp.sum(selected) == n,
            cp.sum(outlier) <= np.ceil(self.outlier_prop * train_n),
            cp.sum(covered, axis=1) == 1,
            cp.multiply(is_far, covered - outlier) == 0
        ]
        for j in range(train_n):
            constraints.append(covered[:, j] <= selected[j])
        for i in range(train_n):
            if self.selected[i]:
                constraints.append(selected[i] == 1)

        prob = cp.Problem(cp.Minimize(1), constraints=constraints)
        prob.solve(solver=cp.GLPK_MI)
        return prob, selected

    def get(self, n: int):

        if self.embedding_mode == 'output':
            # need to train to get proper embedding first
            if isinstance(self.model, JaxNNModel):
                # if self.selected.any():
                #     train_mlp(fn=self.fn,
                #               dataset=self.dataset,
                #               training_subset=self.selected,
                #               training_steps=self.train_epochs,
                #               loss_fn='ce')
                # output = np.array(self.fn(self.dataset.xs_train))
                # TODO: add training for JAX fn as well
                raise ValueError
            elif isinstance(self.model, TorchNNModel):
                if self.use_cuda:
                    self.model.to(self.device)
                assert self.dataset.problem_type == 'classification'
                if self.selected.any():
                    torch_dataset, _ = data_loaders(dataset=self.dataset, 
                                                    selected=self.get_selected(), 
                                                    batch_size=self.train_batch_sz, 
                                                    device=self.device, 
                                                    shuffle=True)
                    train_torch(model=self.model,
                                optimizer=self.optimiser_generator(self.model.parameters()),
                                criterion='ce',
                                train_loader=torch_dataset,
                                epochs=self.train_epochs)
                train_batches, _ = data_loaders(dataset=self.dataset, 
                                                batch_size=self.train_batch_sz, 
                                                device=self.device, 
                                                shuffle=False)
                output = []
                with torch.no_grad():
                    for train_idx, x_train, _ in train_batches:
                        # forward propogation
                        train_pred, _ = self.model.get_penultimate_and_final_output(x_train)
                        train_pred = train_pred.detach().cpu().numpy()
                        output.append(train_pred)
                output = np.concatenate(output, axis=0)
            else:
                assert False
            K = output @ output.T
            cov = np.diag(K)
            cov_rep = np.repeat(cov[:, np.newaxis], cov.shape[0], 1)
            self.dist = np.sqrt(cov_rep + cov_rep.T - 2 * K)
            self._debug['dist_mat'] = self.dist

        m = np.sum(self.selected)
        u = None
        self.last_batch_idxs = []

        if m == 0:
            closest = np.full(self.dataset.train_count(), fill_value=np.inf)
        else:
            closest = np.min(self.dist[:, self.selected], axis=1)

        for i in tqdm.trange(m, n + m):

            if m == 0:
                u = np.random.randint(0, self.dataset.train_count())

            elif i == m:
                u = int(np.argmax(closest))

            else:
                closest = np.min(np.array([closest, self.dist[u]]), axis=0)
                u = int(np.argmax(closest))

            self.last_batch_idxs.append(u)
            self._debug[f'val_round{i}'] = (closest, u)

        if self.do_ilp:
            print('MIP stage...')
            ub = np.max(closest)
            lb = ub / 2.
            test_rads = self.dist[(lb <= self.dist) & (self.dist <= ub)]
            test_rads.sort()
            while ub != lb:
                rad = (lb + ub) / 2.
                prob, selected = self._solve_ilp(n + m, rad)
                print(f'lb={lb:.5f} | rad={rad:.5f} | ub={ub:.5f} | mi_prob_status={prob.status} | '
                      f'poss_rad_count={len(test_rads)}')
                if prob.status == 'optimal':
                    test_rads = test_rads[test_rads <= rad]
                    ub = test_rads[-1]
                else:
                    test_rads = test_rads[test_rads >= rad]
                    lb = test_rads[0]
            prob, selected = self._solve_ilp(n + m, lb)
            self.selected = selected.value.astype(bool)

        else:
            self.selected[self.last_batch_idxs] = True
