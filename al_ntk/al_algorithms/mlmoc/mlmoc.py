import os
import jax
import torch
import time
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
# from torch.utils.tensorboard import SummaryWriter

from typing import Callable, Union
from functools import partial

from al_ntk.dataset import Dataset
from al_ntk.model import JaxNNModel, TorchNNModel, NNModel
from ..base_al_algorithm import ALAlgorithm
from al_ntk.utils.entropy_helper import max_entropy_selector, max_entropy_selector_from_fn
from al_ntk.utils.kernels_helper import compute_kernel_in_batches, approximate_full_kernel_as_block_diag
from al_ntk.utils.maps import optim_map, loss_map
from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.nn_training_torch import train_torch


from .utils import ntk_util, batchnorm_utils  #util, 
from .core.evaluators import AccEvaluator
# from .builders import model_builder, dataloader_builder
from .utils.probability_utils import project_into_probability_simplex, calc_MI_for_pairwise_features

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.80'


class MLMOCMethod(ALAlgorithm):

    def __init__(self, dataset: Dataset, model: NNModel, rand_first_batch: bool = False,
                 train_as_regression: bool = True, sequential_selection: bool = False,
                 train: dict = None, optim: dict = None, use_cuda: bool = True):
        super().__init__(dataset=dataset, model=model)
        self.train_dict = train
        self.optim_dict = optim
        self.rand_first_batch = rand_first_batch
        self.train_as_regression = train_as_regression
        self.sequential_selection = sequential_selection
        # self.logger = util.load_log(name='MLMOC')

        # Load configurations
        # config = util.load_config(config_path)

        # self.model_config = config['model']
        # self.train_config = config['train']
        # self.eval_config = config['eval']
        # self.data_config = config['data']

        # self.eval_standard = self.eval_config['standard']

        # Determine which device to use
        device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()

        if device == 'cpu':
            print('GPU is not available.')
        else:
            print('GPU is available with {} devices.'.format(self.num_devices))
        print('CPU is available with {} devices.'.format(jax.device_count('cpu')))
        
        self.cycle = 0  # self.train_config.get('cycles', 10)
        self._build(mode='train', init=True)

    def _build(self, mode, init=False):

        # Build a dataloader
        # self.dataloaders = dataloader_builder.build(
        #     self.data_config, self.logger)

        # Build an optimizer, scheduler and criterion
        self.optim_gen = lambda params: optim_map[self.optim_dict['name']](params, **self.optim_dict['params'])
        self.optimizer = self.optim_gen(self.model.parameters())
        # self.scheduler = scheduler_builder.build(
        #     self.train_config, self.optimizer, self.logger,
        #     self.train_config['num_epochs'], len(self.dataloaders['train']))
        self.criterion = loss_map['regression']() if self.train_as_regression else loss_map[self.dataset.problem_type]()
        # self.loss_meter, self.pr_meter = meter_builder.build(
        #     self.model_config, self.logger)
        # self.evaluator = evaluator_builder.build(self.eval_config, self.logger)
        # self.evaluator = AccEvaluator(self.logger)

    # def run(self):
    #     trials = self.train_config.get('trials', 1)
    #     cycles = self.train_config.get('cycles', 10)
    #     query_first = self.train_config.get('query_first', False)

    #     for trial in range(trials):
    #         # Build components
    #         init = True if trial == 0 else False
    #         self._build(mode='train', init=init)

    #         for cycle in range(cycles):
    #             if query_first:
    #                 self._update_data(trial, cycle)

    #             # Train a model
    #             self._train(trial, cycle)

    #             # Query new data points and update labeled and unlabeled pools
    #             if not query_first:
    #                 self._update_data(trial, cycle)
    
    def get(self, n: int):
        
        if self.rand_first_batch and (self.cycle == 0):
            print('First batch is generated randomly.')
            rand_idxs = np.random.permutation(np.arange(self.dataset.train_count()))
            count = 0
            for i in rand_idxs:
                if not self.selected[i]:
                    self.selected[i] = True
                    count += 1
                if count == n:
                    break
        
        else:
            
            if self.selected.any():
                # Train a model
                self._train(trial=0, cycle=self.cycle)
            else:
                # pick at least one point randomly to make thing not crash
                self.selected[np.random.choice(self.dataset.train_count())] = True
                n = n - 1

            # Query new data points and update labeled and unlabeled pools
            self._update_data(trial=0, cycle=self.cycle, n=n)
            
        self.cycle += 1


    def _train(self, trial, cycle):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_dict['epochs']  #self.train_config.get('num_epochs', 200)
        self.optimizer = self.optim_gen(self.model.parameters())
        self.model.init_weights()

        print(
            'Trial {}, Cycle - {} - train for {} epochs starting from epoch {}'.format(
                trial, cycle, num_epochs, start_epoch))

        # if self.train_config.get('manual_train_control', False):
        #     print('Training...')
        #     import IPython; IPython.embed()

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps)
            train_time = time.time() - train_start

            # lr = self.scheduler.get_lr()[0]

            # self.logger.infov(
            #     '[Cycle {}, Epoch {}] completed in {:3f} - train loss: {:4f}'\
            #     .format(cycle, epoch, train_time, self.loss_meter.avg))
            # self.writer.add_scalar('Train/learning_rate', lr, global_step=num_steps)

            # if not self.train_config['lr_schedule']['name'] in ['onecycle']:
            #     self.scheduler.step()

            # self.loss_meter.reset()

            # if epoch - start_epoch > 0.8 * num_epochs:
            #     is_last_epoch = start_epoch + num_epochs == epoch + 1
            #     eval_metrics = self._evaluate_once(trial, cycle, epoch, num_steps, is_last_epoch)
            #     self.logger.info(
            #         '[Epoch {}] - {}: {:4f}'.format(
            #             epoch, self.eval_standard, eval_metrics[self.eval_standard]))

        # if self.train_config.get('adjust_batchnorm_stats', True):
        #     self._adjust_batchnorm_to_population()
        #     eval_metrics = self._evaluate_once(trial, cycle, start_epoch + num_epochs, num_steps, True)
        #     self.logger.info(
        #         'After BN adjust - {}: {:4f}'.format(self.eval_standard, eval_metrics[self.eval_standard]))

    def _train_one_epoch(self, epoch, num_steps):

        self.model.train()
        # dataloader = self.dataloaders['train']
        dataloader, _ = data_loaders(
            dataset=self.dataset, 
            selected=self.get_selected(), 
            batch_size=self.train_dict['data_loader_args']['batch_size'], 
            shuffle=True,
            in_class_val=1.,
            not_in_class_val=0.,
            generate_reg_data=self.train_as_regression,
            device=self.device
        )
        num_batches = len(dataloader)

        # for i, input_dict in enumerate(dataloader):
        #     input_dict = util.to_device(input_dict, self.device)

        #     # Forward propagation
        #     self.optimizer.zero_grad()
        #     output_dict = self.models['model'](input_dict)

        #     # Compute losses
        #     output_dict['labels'] = input_dict['labels']

        #     losses = self.criterion(output_dict)
        #     loss = losses['loss']

        #     # Backward propagation
        #     loss.backward()
        #     self.optimizer.step()

        #     # Print losses
        #     batch_size = input_dict['inputs'].size(0)
        #     # self.loss_meter.update(loss.item(), batch_size)
        #     # if i % (len(dataloader) / 10) == 0:
        #     #     self.loss_meter.print_log(epoch, i, num_batches)

        #     # step scheduler if needed
        #     # if self.train_config['lr_schedule']['name'] in ['onecycle']:
        #     #     self.scheduler.step()
                
        # train for one epoch, collect the labels, predictions and losses, and log them
        self.model.train()
        
        for train_idx, x_train, train_label in dataloader:
            self.optimizer.zero_grad()
            # forward propogation
            train_pred = self.model(x_train)
            # calculate loss
            train_loss = self.criterion(train_pred, train_label)
            # back prop
            train_loss.backward()
            self.optimizer.step()

            # Save a checkpoint
            num_steps += x_train.size(0)

        return num_steps

    # def _evaluate_once(self, trial, cycle, epoch, num_steps, is_last_epoch=False):
    #     dataloader = self.dataloaders['val']

    #     self.model.eval()
    #     self.logger.info('[Cycle {} Epoch {}] Evaluating...'.format(cycle, epoch))
    #     labels = []
    #     outputs = []

    #     for input_dict in dataloader:
    #         with torch.no_grad():
    #             input_dict = util.to_device(input_dict, self.device)
    #             # Forward propagation
    #             output_dict = self.models['model'](input_dict)
    #             output_dict['labels'] = input_dict['labels']
    #             labels.append(input_dict['labels'])
    #             outputs.append(output_dict['logits'])

    #     output_dict = {
    #         'logits': torch.cat(outputs),
    #         'labels': torch.cat(labels)
    #     }

    #     if is_last_epoch and False:
    #         probs = project_into_probability_simplex(output_dict['logits'].detach().cpu().numpy())
    #         mis = calc_MI_for_pairwise_features(probs)
    #         print('Mutual information table:')
    #         print(pd.DataFrame(mis))
    #         print('Norm of them:', np.linalg.norm(mis))

    #     # Print losses
    #     self.evaluator.update(output_dict)

    #     self.evaluator.print_log(epoch, num_steps)
    #     eval_metric = self.evaluator.compute()

    #     # Reset the evaluator
    #     self.evaluator.reset()
    #     return {self.eval_standard: eval_metric}

    # def _adjust_batchnorm_to_population(self):

    #     self.logger.info('Adjusting BatchNorm statistics to population values...')

    #     net = self.model
    #     train_dataset = self.dataloaders['train'].dataset
    #     trainloader = torch.utils.data.DataLoader(train_dataset,
    #                                              batch_size=self.data_config['batch_size'],
    #                                              num_workers=self.data_config['num_workers'],
    #                                              drop_last=True)

    #     net.apply(batchnorm_utils.adjust_bn_layers_to_compute_populatin_stats)
    #     for _ in range(3):
    #         with torch.no_grad():
    #             for input_dict in trainloader:
    #                 input_dict = util.to_device(input_dict, self.device)
    #                 net(input_dict)
    #     net.apply(batchnorm_utils.restore_original_settings_of_bn_layers)

    #     self.logger.info('BatchNorm statistics adjusted.')

    def _update_data(self, trial, cycle, n):
        # al_params = self.data_config['al_params']
        al_params = {'add_num': n}
        
        data_pool, _ = data_loaders(
            dataset=self.dataset, 
            selected=None, 
            batch_size=self.train_dict['data_loader_args']['batch_size'], 
            shuffle=False,
            generate_reg_data=False,
            device=self.device
        )
        
        # labelled, _ = data_loaders(
        #     dataset=self.dataset, 
        #     selected=self.get_selected(), 
        #     batch_size=self.train_dict['data_loader_args']['batch_size'], 
        #     shuffle=False,
        #     generate_reg_data=False,
        #     device=self.device
        # )

        # checkpoint_path = os.path.join(self.save_dir, 'checkpoint_tmp.pth')
        # model_params = {'trial': trial}
        # model_params['state_dict'] = self.model.state_dict()
        # torch.save(model_params, checkpoint_path)
        # del model_params

        #### BEGIN NTK ####
        
        bn_with_running_stats = True  # self.model_config['model_arch'].get('bn_with_running_stats', True)
        # self.models['ntk_params'] = ntk_util.update_ntk_params(
        #     self.models['ntk_params'], self.models['model'], bn_with_running_stats)

        # subset_predictions = self._forward_pass()
        unlabeled_dataset = data_pool.dataset
        data = np.arange(len(unlabeled_dataset))
        X, _ = ntk_util.get_full_data(unlabeled_dataset, data)
        batch_size = X.shape[0]
        predictions = []
        with torch.no_grad():
            for i in range(0, batch_size, 1000):
                pred = self.model(X[i:i+1000].to(self.device)).detach().cpu().numpy()
                predictions.append(pred)
        subset_predictions = np.concatenate(predictions)

        # del self.optimizer
        # del self.scheduler
        torch.cuda.empty_cache()

        device_count = self.num_devices if torch.cuda.device_count() > 0 else 1
        cycle_count = 10  #self.train_config.get('cycles', 10)
        
        ntk_config = {
            'diag_reg': 1e-5,  # ntk_config.get('ntk_diag_reg', 1e-5)
            'diag_reg_per_cycle': 1e-5,  # ntk_config.get('ntk_diag_reg_pc', 1e-5)
            'diag_reg_starting_cycle': 5,
            'ntk_batch': 256, 
            't': 32, 
            'lr': self.optim_dict['params']['lr'],
            'momentum': self.optim_dict['params'].get('momentum', 0.),
            'eps_end': 0.,  # ntk_config.get('eps_end', 0.0)
            'eps_method': 'poly',  # ntk_config.get('eps_method', 'poly')
            'ntk_objective': 'pseudo_contrastive',  # ntk_config.get('ntk_objective', 'pseudo_contrastive')
            'use_dpp': False,  # ntk_config.get('dpp', False)
            'dynamic_ntk_batch_coef': 12,  # 
            'ntk_data_noise': 0.,
            'sigmoid_temperature': 4.,
            'softmax_temperature': 1.,
            'pseudo_label_strategy': 'torch',
            'sequential_selection': self.sequential_selection,
            'sequential_with_true_label': False,
        }

        # Measure uncertainty of each data points in the subset
        uncertainty = ntk_util.get_uncertainty_ntk_block_sequential(
            data_pool=data_pool, selected=self.selected,
            model=self.model, subset_predictions=subset_predictions, al_params=al_params,
            cycle=cycle, cycle_count=cycle_count, ntk_config=ntk_config, sample_count=n, num_devices=device_count)

        arg = uncertainty
        
        #### END NTK ####

        # to not select points that are selected already
        count = 0
        for i in arg[::-1]:
            if not self.selected[i]:
                self.selected[i] = True
                count += 1
            if count == n:
                break
            
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        # if isinstance(self.models['model'], torch.nn.DataParallel):
        #     self.models['model'].module.load_state_dict(checkpoint, strict=True)
        # else:
        #     self.models['model'].load_state_dict(checkpoint, strict=True)

        # Create a new dataloader for the updated labeled dataset
        # self.dataloaders = dataloader_builder.update(
        #     cycle, self.dataloaders, arg, self.data_config, self.model_config, self.writer, self.save_dir)

        # self.optimizer = self.optim_gen(self.model.parameters())
        # self.optimizer = optimizer_builder.build(
        #     self.train_config['optimizer'], self.models['model'].parameters(), self.logger)
        # self.scheduler = scheduler_builder.build(
        #     self.train_config, self.optimizer, self.logger, self.train_config['num_epochs'],
        #     len(self.dataloaders['train']))
