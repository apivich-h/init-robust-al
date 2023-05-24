N_THREADS = 4

import os
import traceback

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

os.environ["OMP_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["OPENBLAS_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["MKL_NUM_THREADS"] = f'{N_THREADS}' 
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{N_THREADS}' 
os.environ["NUMEXPR_NUM_THREADS"] = f'{N_THREADS}' 

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import jax
import jax.config
import torch

torch.set_num_threads(N_THREADS)

# only run JAX on cpu
jax.config.update('jax_platform_name', 'cpu')

# set precision for jax
jax.config.update("jax_enable_x64", True)

# set precision for torch
torch.set_default_dtype(torch.float32)

try:
    print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs={jax.local_device_count("gpu")}')
except RuntimeError:
    print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs=None')

print(f'Torch: GPU_available={torch.cuda.is_available()}')


from al_ntk.experiments.utils import init_dir, get_kwargs
from al_ntk.experiments.log import Logger, Timer
from al_ntk.utils.maps import model_map
from al_ntk.experiments.al_utils.maps import data_map, al_map


def active_learning(cfg: DictConfig, logger: Logger, timer: Timer):

    # dataset class or function, queried from map in conf/config.py
    Dataset = data_map[cfg.data.name]
    # dataset object initialized using configurations from conf/data/*.yaml
    if cfg.data.name in {'test', 'mock-mismatch'}:
        Model = model_map[cfg.model.name]['jax']
        fn = Model(in_dim=cfg.data.model_gen.in_dim, out_dim=cfg.data.model_gen.out_dim,
                   **get_kwargs(cfg.model.params), **get_kwargs(cfg.model.jax))
        data = Dataset(**get_kwargs(cfg.data.params, update={'fn': fn}))
    else:
        data = Dataset(**get_kwargs(cfg.data.params))
    # save the dataset to train with later
    logger.save(obj=data, fn='data')

    # fn class to perform AL with, initialized using configurations from conf/fn/*.yaml
    if cfg.al.alg.use_torch:
        Model = model_map[cfg.model.name]['torch']
        model = Model(in_dim=data.input_dimensions(), out_dim=data.output_dimensions(), use_cuda=cfg.use_cuda,
                      **get_kwargs(cfg.model.params), **get_kwargs(cfg.model.torch))
        if cfg.al.alg.to_train:
            with open_dict(cfg):
                cfg.al.alg.kwargs = OmegaConf.merge(cfg.al.alg.kwargs, {'train': OmegaConf.to_container(cfg.train), 
                                                                        'optim': OmegaConf.to_container(cfg.train.optim), 
                                                                        'use_cuda': cfg.use_cuda})
    else:
        Model = model_map[cfg.model.name]['jax']
        model = Model(in_dim=data.input_dimensions(), out_dim=data.output_dimensions(), 
                      **get_kwargs(cfg.model.params), **get_kwargs(cfg.model.jax))

    # timing the initialization of the AL algorithm 
    al_alg = timer.timeit(method=al_map[cfg.al.alg.name], args=tuple((data, model)), kwargs=get_kwargs(cfg.al.alg.kwargs), log_str='initializing AL algorithm')

    total_count = np.arange(0, min(data.train_count(), cfg.data.budget) + 1, cfg.data.batch_sz)
    # additional number of points that need to be queried in every iteration 
    query_count = total_count[1:] - (0 if cfg.al.alg.restart_iter else total_count[:-1])

    logger.log(f'Experiments using AL with {cfg.al.alg.name}')

    for total, to_get in zip(total_count[1:].astype(int), query_count):

        # directory with {size} points selected using AL
        size_dir = f"al_splits/size={str(total).rjust(len(str(total_count[-1])), '0')}"

        # reinitialize AL if gotta
        if cfg.al.alg.restart_iter:
            al_alg = timer.timeit(method=al_map[cfg.al.alg.name], args=tuple((data, model)), kwargs=get_kwargs(cfg.al.alg.kwargs), log_str='reinitializing AL algorithm')
        
        # get the required number of data points 
        logger.log(f'Selecting {to_get} samples from the total data, total pool size of {total} points.')
        _ = timer.timeit(method=al_alg.get, args=tuple((to_get,)), log_str=f'selecting {to_get} points')
        
        # get the selected data points and save them
        selected = al_alg.get_selected()
        logger.save(obj=selected, fn=f'{size_dir}/selected_points')


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:

    # results/active_learning/date-time_data_alg_model
    exp_par_dir = os.path.join('_'.join((cfg.results_dir, cfg.al.alg.name_str, cfg.model.name_str)), 'active_learning')
    print(f'Saving active learning logs at {exp_par_dir}')
    log_fn = os.path.join(exp_par_dir, 'logs')

    # initialize logger and timer objects
    logger = Logger(exp_dir=exp_par_dir, log_fn=log_fn)
    timer = Timer(logger=logger)

    # save configuration yaml file
    logger.log(f'Starting experiment with the following configurations -\n{logger.indent(OmegaConf.to_yaml(cfg).rstrip(), indent=1)}')
    with open(os.path.join(exp_par_dir, 'cfg.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    try:
        active_learning(cfg, logger, timer)
    except Exception as e:
        # log error if run into any
        error = traceback.format_exc().rstrip()
        logger.log(f"Ran into an error -\n{logger.indent(error, indent=1)}")
        print(error)
    
if __name__ == '__main__':

    main()