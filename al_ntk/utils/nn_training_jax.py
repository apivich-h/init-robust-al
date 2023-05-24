import random

import tqdm
from typing import Callable, Union
import numpy as np

import jax.nn
from jax.example_libraries import optimizers
from jax import jit, grad
import jax.numpy as jnp
from jax.tree_util import tree_flatten

from al_ntk.dataset import Dataset
from al_ntk.model import JaxNNModel


def cross_entropy(logits, targets):
    logprobs = jax.nn.log_softmax(logits)
    nll = jnp.take_along_axis(logprobs, targets, axis=1)
    return -jnp.mean(nll)


def binary_cross_entropy(preds, targets):
    logprobs = jax.nn.sigmoid(preds)
    nll = targets * jnp.log(logprobs) + (1 - targets) * jnp.log(1. - logprobs)
    return -jnp.mean(nll)


def mean_squared_error(preds, targets):
    return 0.5 * jnp.mean(jnp.sum((preds - targets)**2, axis=1))


def train_mlp(model: JaxNNModel, dataset: Dataset, training_subset=None, epochs=10000, 
              loss_fn: Union[Callable, str] = 'mse', regularisation_factor: float = 0., batch_sz: int = -1,
              optimiser: str = 'sgd', learning_rate=1e-3,
              prog_bar: bool = False, record_loss_every: int = 100):
    """ Function to train MLPJax
    The training is done using GD (no stochasticity)

    Parameters
    ----------
    model
    dataset
    training_subset: indices to be used for training
    epochs: how many training steps
    learning_rate
    loss_fn: loss function to use - can be mse, ce or bce (mean-squared, cross-entropy, binary cross-entropy)
    regularisation_factor: regularisation constant
    prog_bar: whether to output a progress bar
    record_loss_every: record the train and test loss every x steps

    Returns
    -------

    """
    if optimiser == 'sgd':
        optimiser = optimizers.sgd(learning_rate)
    elif optimiser == 'adam':
        optimiser = optimizers.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    elif optimiser == 'momentum':
        optimiser = optimizers.momentum(learning_rate, mass=0.9)
    else:
        raise ValueError('invalid optimiser')
        
    opt_init, opt_update, get_params = optimiser
    opt_update = jit(opt_update)
    opt_state = opt_init(model.params)

    xs_train, ys_train = dataset.get_train_data()
    xs_test, ys_test = dataset.get_test_data()

    if loss_fn == 'mse':
        fn = mean_squared_error
        ys_train, ys_test = dataset.generate_label_for_regression()
    elif loss_fn == 'ce':
        assert dataset.problem_type == 'classification'
        fn = cross_entropy
    elif loss_fn == 'bce':
        assert dataset.problem_type == 'binary_classification'
        fn = binary_cross_entropy
    else:
        raise ValueError('Invalid loss_fn')

    if regularisation_factor == 0.:
        loss_fn = lambda params, x, y: fn(model.apply_fn(params, x), y)
    elif regularisation_factor > 0.:
        # assert fn.parametrisation == 'ntk'
        leaves_init = [jnp.copy(x) for x in tree_flatten(model.params)[0]]
        def loss_fn(params, x, y):
            leaves, _ = tree_flatten(params)
            reg = sum(jnp.vdot(leaf_1 - leaf_2, leaf_1 - leaf_2)
                      for leaf_1, leaf_2 in zip(leaves, leaves_init))
            return fn(model.apply_fn(params, x), y) + (0.5 * regularisation_factor**2 / xs_train.shape[0]) * reg
    else:
        raise ValueError('regularisation_factor should be >= 0.')

    if training_subset is not None:
        xs_train = xs_train[training_subset, :]
        ys_train = ys_train[training_subset, :]

    loss = jit(loss_fn)
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    record_steps = []
    train_losses = []
    test_losses = []

    for i in (tqdm.trange(epochs) if prog_bar else range(epochs)):

        # construct batches for SGD
        if batch_sz < 1:
            # normal GD
            mini_batches = [None]
        else:
            idxs = list(np.random.permutation(xs_train.shape[0]))
            mini_batches = [idxs[k:k + batch_sz] for k in range(0, xs_train.shape[0], batch_sz)]

        # do GD or SGD
        for batch in mini_batches:
            opt_state = opt_update(i, grad_loss(opt_state, xs_train[batch], ys_train[batch]), opt_state)

        if ((i + 1) % record_loss_every == 0) or ((i + 1) == epochs):
            record_steps.append(i + 1)
            train_losses += [loss(get_params(opt_state), xs_train, ys_train)]
            if xs_test is not None:
                test_losses += [loss(get_params(opt_state), xs_test, ys_test)]
            else:
                test_losses += [jnp.nan]

    model.params = get_params(opt_state)
    return opt_state, optimiser, record_steps, train_losses, test_losses
