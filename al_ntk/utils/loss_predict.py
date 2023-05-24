from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import partial

from sklearn.cluster import KMeans
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp

from al_ntk.dataset import Dataset
from al_ntk.sparse_gp import SparseGP
from al_ntk.sparse_gp.sparse_gp import _test_posterior


@jax.jit
def avg_entropy(variances: jnp.array):
    """
    average log of variances
    this value is proportional to average entropy
    """
    return jnp.mean(jnp.log(variances))


@jax.jit
def _avg_entropy_from_kernel_matrices(qu_cov, K_uu_noise_inv, K_ut, K_tu, K_tt):
    variances = _test_posterior(K_uu_noise_inv=K_uu_noise_inv,
                                K_ut=K_ut,
                                K_tu=K_tu,
                                K_tt=K_tt,
                                qu_cov=qu_cov)
    return avg_entropy(variances)


def grad_avg_entropy_wrt_inducing_cov(sgp: SparseGP, qu_cov: jnp.array):
    """
    del(L) where L is avg_entropy and
    del is wrt covariance of inducing points effective prior
    """
    grad_loss_fn = jax.jit(jax.grad(_avg_entropy_from_kernel_matrices))
    grad_loss = grad_loss_fn(qu_cov=qu_cov,
                             K_uu_noise_inv=sgp.K_uu_inv,
                             K_ut=sgp.K_ut,
                             K_tu=sgp.K_tu,
                             K_tt=sgp.K_tt)
    return grad_loss
