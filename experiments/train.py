N_THREADS = 2

import os
import copy
from tqdm import tqdm
import pickle

os.environ["OMP_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["OPENBLAS_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["MKL_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{N_THREADS}' 
os.environ["NUMEXPR_NUM_THREADS"] = f'{N_THREADS}' 

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

torch.set_num_threads(N_THREADS)

from al_ntk.utils.torch_dataloader import data_loaders
from al_ntk.utils.maps import model_map, loss_map, optim_map
from al_ntk.judge_metric import RMSE, Entropy, NLLH
from al_ntk.experiments.utils import init_dir, get_kwargs
from al_ntk.experiments.log import Logger


# # update the running tally of labels, predictions and losses, indexed, in each epoch
# def update(labels, preds, losses, label, pred, loss, idx):

#     idx = idx.to(dtype=torch.long).reshape(-1, 1)
#     output = []
#     for iterable, elem in zip((labels, preds, losses), (label, pred, loss)):
#         if len(elem.shape) == 1:
#             elem = elem.reshape(-1, 1)
#         output.append(torch.concat((iterable, torch.concat((idx, elem), axis=1)), axis=0))
    
#     return output


# # update the running tally of labels, predictions and losses, indexed, in each epoch
# def update(labels, preds, losses, idxs):

#     labels = torch.cat(labels)
#     preds = torch.cat(preds)
#     losses = torch.cat(losses)
#     idxs = torch.cat(idxs).to(dtype=torch.long).reshape(-1, 1)

#     output = []
#     for iterable in (labels, preds, losses):
#         if len(iterable.shape) == 1:
#             iterable = iterable.reshape(-1, 1)
#         output.append(torch.concat((idxs, iterable), axis=1))
    
#     return output

# update the running tally of labels, predictions and losses, indexed, in each epoch
def update(idxs, *values):
    idxs = torch.cat(idxs).to(dtype=torch.long).reshape(-1, 1)
    outputs = []
    for iterable in values:
        iterable = torch.cat(iterable)
        if len(iterable.shape) == 1:
            iterable = iterable.reshape(-1, 1)
        outputs.append(torch.concat((idxs, iterable), axis=1))
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
    

# log the labels, predictions and losses in specific log files
def log(logger, epoch_dir, iterables, log_fns):

    for iterable, log_fn in zip(iterables, log_fns):
        logger.write(
            text='\n'.join(list(map(lambda x: '\t'.join(list(map(str, x))), iterable.cpu().detach().tolist()))), 
            fn=os.path.join(epoch_dir, log_fn)
        )
  
      
# log the labels, predictions and losses as pipckle files in specific log files
def log_pickle(logger, epoch_dir, iterables, log_fns):

    for iterable, log_fn in zip(iterables, log_fns):
        logger.save(
            obj=iterable.cpu().detach().numpy(), 
            fn=os.path.join(epoch_dir, log_fn)
        )


def train(epochs, train_loader, test_loader, model, optimizer, criterion, logger, device, size_dir, save_every):

    for epoch in range(1, epochs+1):

        epoch_dir = os.path.join(size_dir, f"epoch={str(epoch).rjust(len(str(epochs+1)), '0')}")
        
        # train for one epoch, collect the labels, predictions and losses, and log them
        model.train()
        train_ys, train_preds, train_losses, train_idxs = [], [], [], []
        for train_idx, train_x, train_y in train_loader:
            # reset the optimizer
            optimizer.zero_grad()
            # forward propogation
            train_x.to(device)
            train_pred = model(train_x)
            train_loss = criterion(train_pred, train_y)
            # # update labels, predictions and losses
            # train_ys, train_preds, train_losses = update(train_ys, train_preds, train_losses, train_y, train_pred, train_loss, train_idx)
            # back propogate the average loss
            train_loss_mean = train_loss.mean()
            train_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            # optimizer step
            optimizer.step()
            if len(train_loss.shape) == 2:
                train_loss = train_loss.mean(dim=1)
            train_losses.append(train_loss.cpu())
            train_idxs.append(train_idx.cpu())
            if epoch % save_every == 0:
                train_ys.append(train_y.cpu())
                train_preds.append(train_pred.cpu())
                
        if epoch % save_every == 0:
            # update labels, predictions and losses
            train_ys, train_preds, train_losses = update(train_idxs, train_ys, train_preds, train_losses)
            log_pickle(logger=logger, epoch_dir=epoch_dir, 
                       iterables=(train_ys, train_preds, train_losses), 
                       log_fns=('train_labels', 'train_preds', 'train_losses'))
        else:
            train_losses = update(train_idxs, train_losses)

        if (test_loader is not None) and (epoch % save_every == 0):

            # test for one epoch, collect the labels, predictions and losses, and log them
            with torch.no_grad():
                model.eval()
                # test_ys, test_preds, test_losses = torch.Tensor(), torch.Tensor(), torch.Tensor()
                test_ys, test_preds, test_losses, test_idxs = [], [], [], []
                for test_idx, test_x, test_y in test_loader:
                    # forward propogation
                    test_x.to(device)
                    test_pred = model(test_x)
                    test_loss = criterion(test_pred, test_y)
                    if len(test_loss.shape) == 2:
                        test_loss = test_loss.mean(dim=1)
                    if epoch % save_every == 0:
                        test_losses.append(test_loss.cpu())
                        test_idxs.append(test_idx.cpu())
                        test_ys.append(test_y.cpu())
                        test_preds.append(test_pred.cpu())
                        
                # update labels, predictions and losses
                test_ys, test_preds, test_losses = update(test_idxs, test_ys, test_preds, test_losses)
                log_pickle(logger=logger, epoch_dir=epoch_dir, 
                           iterables=(test_preds, test_losses), 
                           log_fns=('test_preds', 'test_losses'))
        
        # log the average metrics
        logger.log(f'Epoch {epoch}: Training Loss = {train_losses[:, 1].mean()}')
        if (epoch % save_every == 0) and (test_loader is not None):
            logger.log(f'Epoch {epoch}: Validation Loss = {test_losses[:, 1].mean()}')
           
        # only save true test label on the last epoch (to save memory) 
        if epoch == epochs:
            log_pickle(logger=logger, epoch_dir=epoch_dir, iterables=(test_ys,), log_fns=('test_labels',))
        

# @hydra.main(version_base=None, config_name='cfg')
# def main(cfg: DictConfig) -> None:

#     # initialize the log directory for training - results/exp_date/exp_time/exp_name/training
#     al_dir_abs = HydraConfig.get().runtime.config_sources[1].path
#     al_dir_rel = os.path.relpath(al_dir_abs)
#     exp_par_dir = os.path.join(al_dir_rel.rsplit('/', 1)[0], 'training')

def train_wrapper(al_dir_rel: str, exp_par_dir: str, data, cfg: DictConfig, Loss, generate_reg_data: bool) -> None:

    print(f'Saving training logs at {exp_par_dir}')
    log_fn = os.path.join(exp_par_dir, 'logs')

    # initialize logger and timer objects
    logger = Logger(exp_dir=exp_par_dir, log_fn=log_fn)

    # log the configurations
    logger.log(f'Starting experiment with the following configurations -\n{logger.indent(OmegaConf.to_yaml(cfg).rstrip(), indent=1)}')
    
    # # load the data
    # with open(f'{al_dir_rel}/data.pickle', 'rb') as f:
    #     data = pickle.load(f)

    # torch fn class to train with
    Model = model_map[cfg.model.name]['torch']
    # fn initialized using configurations from conf/fn/*.yaml
    model = Model(in_dim=data.input_dimensions(), out_dim=data.output_dimensions(), 
                  **get_kwargs(cfg.model.params), **get_kwargs(cfg.model.torch))
    
    # optimizer class
    Optimizer = optim_map[cfg.train.optim.name]
    # loss class - inferred from problem type as in data
    # Loss = loss_map[data.problem_type]
    criterion = Loss(reduction='none')

    # training device configurations
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.log('Using GPU' if use_cuda else 'GPU not found')
    device_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else dict()

    # al sizes
    size_dirs = sorted(os.listdir(f'{al_dir_rel}/al_splits'))

    for iteration in tqdm(range(1, cfg.train.iters+1)):
        iter_dir = f"iteration={str(iteration).rjust(len(str(cfg.train.iters+1)), '0')}"

        # initialize weights
        model.init_weights()

        for size_dir in size_dirs:
            size = size_dir.split('=')[1], 

            # load the points selected by AL
            with open(f'{al_dir_rel}/al_splits/{size_dir}/selected_points.pickle', 'rb') as f:
                selected = pickle.load(f)
            logger.log(f'Training using {size} data points')

            # get data loaders
            train_loader, test_loader = data_loaders(
                dataset=data, 
                selected=selected,
                device=device,
                generate_reg_data=generate_reg_data,
                **cfg.train.data_loader_args
            )

            # copy the fn - use the same init weights to train all the sizes
            model_copy = copy.deepcopy(model)
            model_copy.to(device)

            train(
                epochs=cfg.train.epochs,
                train_loader=train_loader, 
                test_loader=test_loader,
                model=model_copy, 
                optimizer=Optimizer(model_copy.parameters(), **get_kwargs(cfg.train.optim.params)),
                criterion=criterion,
                logger=logger,
                device=device,
                size_dir=os.path.join(iter_dir, size_dir),
                save_every=cfg.train.save_every
            )
            
            
@hydra.main(version_base=None, config_name='cfg')
def main(cfg: DictConfig) -> None:
    
    al_dir_abs = HydraConfig.get().runtime.config_sources[1].path
    al_dir_rel = os.path.relpath(al_dir_abs)
    
    # load the data
    with open(f'{al_dir_rel}/data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    # ==== FIRST TRAIN ROUND - using the "usual" loss function ====
    
    # initialize the log directory for training - results/exp_date/exp_time/exp_name/training
    exp_par_dir_probtype = os.path.join(al_dir_rel.rsplit('/', 1)[0], 'training')
    # loss class - inferred from problem type as in data
    Loss_probtype = loss_map[data.problem_type]
    
    train_wrapper(
        al_dir_rel=al_dir_rel,
        exp_par_dir=exp_par_dir_probtype, 
        data=data, 
        cfg=cfg, 
        Loss=Loss_probtype, 
        generate_reg_data=False
    )
    
    # # ==== SECOND TRAIN ROUND - using MSE loss ====
    
    # if data.problem_type != 'regression':
    
    #     # initialize the log directory for training - results/exp_date/exp_time/exp_name/training
    #     exp_par_dir_mse = os.path.join(al_dir_rel.rsplit('/', 1)[0], 'training-mse')
    #     # loss class - inferred from problem type as in data
    #     Loss_MSE = loss_map['regression']
        
    #     train_wrapper(
    #         al_dir_rel=al_dir_rel,
    #         exp_par_dir=exp_par_dir_mse, 
    #         data=data, 
    #         cfg=cfg, 
    #         Loss=Loss_MSE, 
    #         generate_reg_data=True
    #     )


if __name__ == '__main__':
    main()