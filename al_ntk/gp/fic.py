from typing import Callable, Optional, Union
from functools import partial
import time

import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp

from al_ntk.dataset import Dataset
from al_ntk.gp.sparse_gp import SparseGP
from al_ntk.utils.kernels_helper import compute_diag_kernel_in_batches
from al_ntk.utils.linalg_jax import symmetric_matrix_sum_inverse, diag_matmul


class FIC(SparseGP):

    def __init__(self, kernel_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray], Optional[bool]], jnp.ndarray],
                 Xn: jnp.ndarray, yn: jnp.ndarray, Xt: jnp.ndarray = None, yt: jnp.ndarray = None,
                 sigma_noise: float = 1e-2, keep_training_kernel_vals: bool = False, 
                 inducing_pts: Union[str, jnp.ndarray] = 'kmeans', m: int = 50,
                 use_train_set_for_test: Union[bool, jnp.ndarray] = False, only_track_diag: bool = True,):
        super().__init__(
            kernel_fn=kernel_fn,
            Xn=Xn,
            yn=yn,
            Xt=Xt,
            yt=yt,
            sigma_noise=sigma_noise,
            inducing_pts=inducing_pts,
            m=m,
            use_train_set_for_test=use_train_set_for_test,
            keep_training_kernel_vals=keep_training_kernel_vals,
            only_track_diag=only_track_diag
        )
        self.Q_nn_diag = diag_matmul(self.K_nu, self.K_uu_inv @ self.K_nu.transpose())
        self.Gamma_nn_diag = self.K_nn_diag - self.Q_nn_diag + self.sigma_noise ** 2
        self.zeros_u = jnp.zeros(shape=(self.m, self.yn.shape[1]))
        self.F_batch_all, self.G_batch_all = _fic_F_and_G_batch(K_uu_inv=self.K_uu_inv,
                                                                K_bu=self.K_nu,
                                                                G_bb=self.Gamma_nn_diag,
                                                                yb=self.yn)

        # self.K_dd_diag = None
        # self.Q_dd = None

        self.update_train(jnp.arange(0))

    def _internal_update_train(self, indexs: jnp.ndarray):
        # self.K_dd_diag = self.K_nn_diag[self.indexs]
        # self.Q_dd = self.K_du @ self.K_uu_inv @ self.K_ud
        pass

    def _internal_incrementally_update_train(self, additional_indexs: jnp.ndarray):
        # self.K_dd_diag = jnp.block([self.K_dd_diag, self.K_nn_diag[additional_indexs]])
        # self.Q_dd = self.K_du @ self.K_uu_inv @ self.K_ud
        pass

    def _internal_update_labels(self):
        self.F_batch_all, self.G_batch_all = _fic_F_and_G_batch(K_uu_inv=self.K_uu_inv,
                                                                K_bu=self.K_nu,
                                                                G_bb=self.Gamma_nn_diag,
                                                                yb=self.yn)

    def get_dL_extra_separate(self, idxs: jnp.ndarray = None):
        F_extra, G_extra = self.get_F_and_G_batch(idxs)
        return G_extra - F_extra @ self.qu_mean, 0.5 * F_extra

    def get_dL_extra(self, idxs: jnp.ndarray = None):
        dL_mean_extra, dL_cov_extra = self.get_dL_extra_separate(idxs=idxs)
        return jnp.sum(dL_mean_extra, axis=0), jnp.sum(dL_cov_extra, axis=0)

    def get_dL_extra_separate_nat_param(self, idxs: jnp.ndarray = None):
        F_batch, G_batch = self.get_F_and_G_batch(idxs)
        return G_batch, -0.5 * F_batch

    def get_dL_extra_nat_param(self, idxs: jnp.ndarray = None):
        dL_t1_extra, dL_t2_extra = self.get_dL_extra_separate_nat_param(idxs=idxs)
        return jnp.sum(dL_t1_extra, axis=0), jnp.sum(dL_t2_extra, axis=0)

    def get_F_prime(self):
        return self.K_uu_inv

    def get_G_prime(self):
        return self.zeros_u

    def get_F_and_G_batch(self, idxs: jnp.ndarray = None):
        if idxs is None:
            return self.F_batch_all, self.G_batch_all
        else:
            return self.F_batch_all[idxs, :, :], self.G_batch_all[idxs, :, :]

    def get_updated_inducing_distribution(self, idxs: jnp.ndarray):
        if idxs.shape[0] == 1:
            # for FIC I can probably do it faster than other SGP
            return _incremental_update_mean_and_cov_one(cov_init=self.qu_cov,
                                                        t1=self.qu_theta1 + self.G_batch_all[idxs, :, :],
                                                        K_uu_inv=self.K_uu_inv,
                                                        K_bu=self.K_nu[idxs, :],
                                                        G_bb=self.Gamma_nn_diag[idxs])
        else:
            # for more than one indices, just do it as above
            F_extra, G_extra = self.get_F_and_G_batch(idxs)
            t1, t2 = self.qu_theta1 + jnp.sum(G_extra, axis=0), self.qu_theta2 - 0.5 * jnp.sum(F_extra, axis=0)
            return self.get_mean_and_cov(theta_1=t1, theta_2=t2)


@jax.jit
def _incremental_update_mean_and_cov_one(cov_init, t1, K_uu_inv, K_bu, G_bb):
    B = K_uu_inv @ K_bu.T
    D_inv = -2. * G_bb.reshape(1, 1)
    t2_inv = -2. * cov_init
    cov_updated = -0.5 * symmetric_matrix_sum_inverse(A_inv=t2_inv, B=B, D_inv=D_inv)
    mean_updated = cov_updated @ t1
    return mean_updated, cov_updated


@jax.jit
def _fic_F_and_G_batch(K_uu_inv, K_bu, G_bb, yb):
    a = K_bu @ K_uu_inv
    F_batch = a[:, :, None] * ((1. / G_bb).reshape(-1, 1) * a)[:, None, :]
    G_batch = a[:, :, None] * ((1. / G_bb).reshape(-1, 1) * yb)[:, None, :]
    return F_batch, G_batch
