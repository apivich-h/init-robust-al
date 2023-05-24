# import os
# import traceback

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# import numpy as np
# import hydra
# from omegaconf import DictConfig, OmegaConf, open_dict
# import jax
# import jax.config
# import torch
# from tqdm import tqdm

# from al_ntk.experiments.utils import init_dir, get_kwargs
# from al_ntk.experiments.log import Logger, Timer
# from al_ntk.utils.maps import model_map
# from al_ntk.experiments.al_utils.maps import data_map, al_map
# from al_ntk.model.ensemble import EnsembleModelTorch
# from al_ntk.utils.torch_dataloader import data_loaders
# from al_ntk.utils.maps import model_map, loss_map, optim_map
# from al_ntk.experiments.utils import init_dir, get_kwargs

# # only run JAX on cpu
# jax.config.update('jax_platform_name', 'cpu')

# # set precision for jax
# jax.config.update("jax_enable_x64", True)

# # set precision for torch
# torch.set_default_dtype(torch.float32)

# try:
#     print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs={jax.local_device_count("gpu")}')
# except RuntimeError:
#     print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs=None')

# print(f'Torch: GPU_available={torch.cuda.is_available()}')


# # update the running tally of labels, predictions and losses, indexed, in each epoch
# def update(idxs, *values):
#     idxs = torch.cat(idxs).to(dtype=torch.long).reshape(-1, 1)
#     outputs = []
#     for iterable in values:
#         iterable = torch.cat(iterable)
#         if len(iterable.shape) == 1:
#             iterable = iterable.reshape(-1, 1)
#         outputs.append(torch.concat((idxs, iterable), axis=1))
#     if len(outputs) == 1:
#         return outputs[0]
#     else:
#         return outputs
    

# # log the labels, predictions and losses in specific log files
# def log(logger, epoch_dir, iterables, log_fns):

#     for iterable, log_fn in zip(iterables, log_fns):
#         logger.write(
#             text='\n'.join(list(map(lambda x: '\t'.join(list(map(str, x))), iterable.cpu().detach().tolist()))), 
#             fn=os.path.join(epoch_dir, log_fn)
#         )
  
      
# # log the labels, predictions and losses as pipckle files in specific log files
# def log_pickle(logger, epoch_dir, iterables, log_fns):

#     for iterable, log_fn in zip(iterables, log_fns):
#         logger.save(
#             obj=iterable.cpu().detach().numpy(), 
#             fn=os.path.join(epoch_dir, log_fn)
#         )


# def train(epochs, train_loader, test_loader, model, optimizer, criterion, logger, device, size_dir, save_every):

#     for epoch in range(1, epochs+1):

#         epoch_dir = os.path.join(size_dir, f"epoch={str(epoch).rjust(len(str(epochs+1)), '0')}")
        
#         # train for one epoch, collect the labels, predictions and losses, and log them
#         model.train()
#         train_ys, train_preds, train_losses, train_idxs = [], [], [], []
#         for train_idx, train_x, train_y in train_loader:
#             # reset the optimizer
#             optimizer.zero_grad()
#             # forward propogation
#             train_x.to(device)
#             train_pred = model(train_x)
#             train_loss = criterion(train_pred, train_y)
#             # # update labels, predictions and losses
#             # train_ys, train_preds, train_losses = update(train_ys, train_preds, train_losses, train_y, train_pred, train_loss, train_idx)
#             # back propogate the average loss
#             train_loss_mean = train_loss.mean()
#             train_loss_mean.backward()
#             # optimizer step
#             optimizer.step()
#             if len(train_loss.shape) == 2:
#                 train_loss = train_loss.mean(dim=1)
#             train_losses.append(train_loss.cpu())
#             train_idxs.append(train_idx.cpu())
#             if epoch % save_every == 0:
#                 train_ys.append(train_y.cpu())
#                 train_preds.append(train_pred.cpu())
                
#         if epoch % save_every == 0:
#             # update labels, predictions and losses
#             train_ys, train_preds, train_losses = update(train_idxs, train_ys, train_preds, train_losses)
#             log_pickle(logger=logger, epoch_dir=epoch_dir, 
#                        iterables=(train_ys, train_preds, train_losses), 
#                        log_fns=('train_labels', 'train_preds', 'train_losses'))
#         else:
#             train_losses = update(train_idxs, train_losses)

#         if (test_loader is not None) and (epoch % save_every == 0):

#             # test for one epoch, collect the labels, predictions and losses, and log them
#             with torch.no_grad():
#                 model.eval()
#                 # test_ys, test_preds, test_losses = torch.Tensor(), torch.Tensor(), torch.Tensor()
#                 test_ys, test_preds, test_losses, test_idxs = [], [], [], []
#                 for test_idx, test_x, test_y in test_loader:
#                     # forward propogation
#                     test_x.to(device)
#                     test_pred = model(test_x)
#                     test_loss = criterion(test_pred, test_y)
#                     if len(test_loss.shape) == 2:
#                         test_loss = test_loss.mean(dim=1)
#                     if epoch % save_every == 0:
#                         test_losses.append(test_loss.cpu())
#                         test_idxs.append(test_idx.cpu())
#                         test_ys.append(test_y.cpu())
#                         test_preds.append(test_pred.cpu())
                        
#                 # update labels, predictions and losses
#                 test_ys, test_preds, test_losses = update(test_idxs, test_ys, test_preds, test_losses)
#                 log_pickle(logger=logger, epoch_dir=epoch_dir, 
#                            iterables=(test_preds, test_losses), 
#                            log_fns=('test_preds', 'test_losses'))
        
#         # log the average metrics
#         logger.log(f'Epoch {epoch}: Training Loss = {train_losses[:, 1].mean()}')
#         if (epoch % save_every == 0) and (test_loader is not None):
#             logger.log(f'Epoch {epoch}: Validation Loss = {test_losses[:, 1].mean()}')
           
#         # only save true test label on the last epoch (to save memory) 
#         if epoch == epochs:
#             log_pickle(logger=logger, epoch_dir=epoch_dir, iterables=(test_ys,), log_fns=('test_labels',))


# def active_learning(cfg: DictConfig, logger_al: Logger, 
#                     logger_training: Logger, logger_training_mse: Logger, 
#                     # logger_training_ens: Logger, logger_training_mse_ens: Logger,
#                     timer: Timer):

#     # dataset class or function, queried from map in conf/config.py
#     Dataset = data_map[cfg.data.name]
#     # dataset object initialized using configurations from conf/data/*.yaml
#     if cfg.data.name in {'test', 'mock-mismatch'}:
#         Model = model_map[cfg.model.name]['jax']
#         fn = Model(in_dim=cfg.data.model_gen.in_dim, out_dim=cfg.data.model_gen.out_dim,
#                    **get_kwargs(cfg.model.params), **get_kwargs(cfg.model.jax))
#         data = Dataset(**get_kwargs(cfg.data.params, update={'fn': fn}))
#     else:
#         data = Dataset(**get_kwargs(cfg.data.params))
#     # save the dataset to train with later
#     logger_al.save(obj=data, fn='data')

#     # fn class to perform AL with, initialized using configurations from conf/fn/*.yaml
#     Model = model_map[cfg.model.name]['torch']
#     model = EnsembleModelTorch.construct_ensemble(
#         in_dim=data.input_dimensions(), 
#         out_dim=data.output_dimensions(), 
#         use_cuda=cfg.use_cuda,
#         **get_kwargs(cfg.model.params)
#     )
#     if cfg.al.alg.to_train:
#         with open_dict(cfg):
#             cfg.al.alg.kwargs = OmegaConf.merge(cfg.al.alg.kwargs, {'train': OmegaConf.to_container(cfg.train), 
#                                                                     'optim': OmegaConf.to_container(cfg.train.optim), 
#                                                                     'use_cuda': cfg.use_cuda})
            
#     print('Ensemble components:')
#     print(model)

#     # timing the initialization of the AL algorithm 
#     al_alg = timer.timeit(method=al_map[cfg.al.alg.name], args=tuple((data, model)), kwargs=get_kwargs(cfg.al.alg.kwargs), log_str='initializing AL algorithm')

#     total_count = np.arange(0, min(data.train_count(), cfg.data.budget) + 1, cfg.data.batch_sz)
#     # additional number of points that need to be queried in every iteration 
#     query_count = total_count[1:] - (0 if cfg.al.alg.restart_iter else total_count[:-1])

#     logger_al.log(f'Experiments using AL with {cfg.al.alg.name}')

#     for total, to_get in zip(total_count[1:].astype(int), query_count):

#         print('=== AL STAGE ===')

#         # directory with {size} points selected using AL
#         size_dir = f"al_splits/size={str(total).rjust(len(str(total_count[-1])), '0')}"

#         # reinitialize AL if gotta
#         if cfg.al.alg.restart_iter:
#             al_alg = timer.timeit(method=al_map[cfg.al.alg.name], args=tuple((data, model)), kwargs=get_kwargs(cfg.al.alg.kwargs), log_str='reinitializing AL algorithm')
        
#         # get the required number of data points 
#         logger_al.log(f'Selecting {to_get} samples from the total data, total pool size of {total} points.')
#         _ = timer.timeit(method=al_alg.get, args=tuple((to_get,)), log_str=f'selecting {to_get} points')
        
#         # get the selected data points and save them
#         selected = al_alg.get_selected()
#         logger_al.save(obj=selected, fn=f'{size_dir}/selected_points')
#         logger_al.save(obj=model.weights, fn=f'{size_dir}/ensemble_lh')
        
#         # optimizer class
#         Optimizer = optim_map[cfg.train.optim.name]

#         # training device configurations
#         use_cuda = cfg.use_cuda and torch.cuda.is_available()
        
#         device = torch.device('cuda' if use_cuda else 'cpu')
#         logger_training.log('Using GPU' if use_cuda else 'GPU not found')
#         logger_training_mse.log('Using GPU' if use_cuda else 'GPU not found')
#         device_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else dict()
        
#         size = total
#         size_dir = f"size={str(total).rjust(len(str(total_count[-1])), '0')}"
        
#         weight_llh = model.weights.clone().cpu()
#         weight_single = torch.nn.functional.one_hot(torch.argmax(weight_llh), num_classes=weight_llh.shape[0]).cpu().type(weight_llh.dtype)
        
#         print(f'weight_llh = {weight_llh}')
#         print(f'weight_single = {weight_single}')

#         print('=== TRAINING STAGE ===')
#         for iteration in tqdm(range(1, cfg.train.iters+1)):
#             iter_dir = f"iteration={str(iteration).rjust(len(str(cfg.train.iters+1)), '0')}"

#             # get data loaders
#             train_loader, test_loader = data_loaders(
#                 dataset=data, 
#                 selected=selected,
#                 device=device,
#                 generate_reg_data=False,
#                 **cfg.train.data_loader_args
#             )

#             # # copy the fn - use the same init weights to train all the sizes
#             # model_copy = copy.deepcopy(model)
#             # model_copy.to(device)

#             logger_training.log(f'Training using {size} data points')
                        
#             # initialize weights
#             model.init_weights()
#             model.reweight_models(weight_single)
#             train(
#                 epochs=cfg.train.epochs,
#                 train_loader=train_loader, 
#                 test_loader=test_loader,
#                 model=model, 
#                 optimizer=Optimizer(model.parameters(), **get_kwargs(cfg.train.optim.params)),
#                 criterion=loss_map[data.problem_type](reduction='none'),
#                 logger=logger_training,
#                 device=device,
#                 size_dir=os.path.join(iter_dir, size_dir),
#                 save_every=cfg.train.save_every
#             )
            
#             # logger_training_ens.log(f'Training using {size} data points')
            
#             # # initialize weights
#             # model.init_weights()
#             # model.reweight_models(weight_llh)
#             # train(
#             #     epochs=cfg.train.epochs,
#             #     train_loader=train_loader, 
#             #     test_loader=test_loader,
#             #     model=model, 
#             #     optimizer=Optimizer(model.parameters(), **get_kwargs(cfg.train.optim.params)),
#             #     criterion=loss_map[data.problem_type](reduction='none'),
#             #     logger=logger_training_ens,
#             #     device=device,
#             #     size_dir=os.path.join(iter_dir, size_dir),
#             #     save_every=cfg.train.save_every
#             # )
            
#             if data.problem_type != 'regression':
                
#                 # get data loaders
#                 train_loader_mse, test_loader_mse = data_loaders(
#                     dataset=data, 
#                     selected=selected,
#                     device=device,
#                     generate_reg_data=True,
#                     **cfg.train.data_loader_args
#                 )
                
#                 logger_training_mse.log(f'Training with MSE using {size} data points')

#                 # initialize weights
#                 model.init_weights()
#                 model.reweight_models(weight_single)
#                 train(
#                     epochs=cfg.train.epochs,
#                     train_loader=train_loader_mse, 
#                     test_loader=test_loader_mse,
#                     model=model, 
#                     optimizer=Optimizer(model.parameters(), **get_kwargs(cfg.train.optim.params)),
#                     criterion=loss_map['regression'](reduction='none'),
#                     logger=logger_training_mse,
#                     device=device,
#                     size_dir=os.path.join(iter_dir, size_dir),
#                     save_every=cfg.train.save_every
#                 )
                
#                 # logger_training_mse_ens.log(f'Training with MSE using {size} data points')

#                 # # initialize weights
#                 # model.init_weights()
#                 # model.reweight_models(weight_llh)
#                 # train(
#                 #     epochs=cfg.train.epochs,
#                 #     train_loader=train_loader_mse, 
#                 #     test_loader=test_loader_mse,
#                 #     model=model, 
#                 #     optimizer=Optimizer(model.parameters(), **get_kwargs(cfg.train.optim.params)),
#                 #     criterion=loss_map['regression'](reduction='none'),
#                 #     logger=logger_training_mse_ens,
#                 #     device=device,
#                 #     size_dir=os.path.join(iter_dir, size_dir),
#                 #     save_every=cfg.train.save_every
#                 # )
            
#         # initialize weights
#         model.init_weights()
#         model.reweight_models(weight_llh)

#         print('---')


# @hydra.main(version_base=None, config_path='./conf', config_name='config')
# def main(cfg: DictConfig) -> None:

#     base_dir = '_'.join((cfg.results_dir, cfg.al.alg.name_str, cfg.model.name_str))

#     # results/active_learning/date-time_data_alg_model
#     exp_par_dir = os.path.join(base_dir, 'active_learning')
#     print(f'Saving active learning logs at {exp_par_dir}')
#     logger_al = Logger(exp_dir=exp_par_dir, log_fn=os.path.join(exp_par_dir, 'logs'))
    
#     tr_par_dir = os.path.join(base_dir, 'training')
#     print(f'Saving training logs at {tr_par_dir}')
#     logger_tr = Logger(exp_dir=tr_par_dir, log_fn=os.path.join(tr_par_dir, 'logs'))
    
#     trmse_par_dir = os.path.join(base_dir, 'training-mse')
#     print(f'Saving training-mse logs at {trmse_par_dir}')
#     logger_tr_mse = Logger(exp_dir=trmse_par_dir, log_fn=os.path.join(trmse_par_dir, 'logs'))
    
#     # tr_par_dir_ens = os.path.join(base_dir, 'training-ens')
#     # print(f'Saving training ens logs at {tr_par_dir_ens}')
#     # logger_tr_ens = Logger(exp_dir=tr_par_dir_ens, log_fn=os.path.join(tr_par_dir_ens, 'logs'))
    
#     # trmse_par_dir_ens = os.path.join(base_dir, 'training-mse-ens')
#     # print(f'Saving training-mse ens logs at {trmse_par_dir_ens}')
#     # logger_tr_mse_ens = Logger(exp_dir=trmse_par_dir_ens, log_fn=os.path.join(trmse_par_dir_ens, 'logs'))

#     # initialize logger and timer objects
#     timer_al = Timer(logger=logger_al)

#     # save configuration yaml file
#     logger_al.log(f'Starting experiment with the following configurations -\n{logger_al.indent(OmegaConf.to_yaml(cfg).rstrip(), indent=1)}')
#     with open(os.path.join(exp_par_dir, 'cfg.yaml'), 'w') as f:
#         f.write(OmegaConf.to_yaml(cfg))

#     try:
#         active_learning(
#             cfg, 
#             logger_al=logger_al, 
#             logger_training=logger_tr,
#             logger_training_mse=logger_tr_mse,
#             # logger_training_ens=logger_tr_ens,
#             # logger_training_mse_ens=logger_tr_mse_ens,
#             timer=timer_al
#         )
#     except Exception as e:
#         # log error if run into any
#         error = traceback.format_exc().rstrip()
#         logger_al.log(f"Ran into an error -\n{logger_al.indent(error, indent=1)}")
#         print(error)
    
# if __name__ == '__main__':

#     main()