from typing import Callable, Optional, Iterable, Union

import jax
import jax.numpy as jnp

from al_ntk.dataset import Dataset
from al_ntk.gp.gp import GP
from al_ntk.utils.linalg_jax import symmetric_block_inverse, symmetric_block_inverse_as_blocks_difference, diag_matmul


class FullRankGP(GP):

    def __init__(self, kernel_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray], Optional[bool]], jnp.ndarray],
                 Xn: jnp.ndarray, yn: jnp.ndarray, Xt: jnp.ndarray = None, sigma_noise: float = 1e-2,
                 use_train_set_for_test: Union[bool, jnp.ndarray] = False, kernel_batch_sz: int = 256, 
                 only_track_diag: bool = False, keep_training_kernel_vals: bool = False, prior_mean_fn: Callable = None):
        super().__init__()
        self.kernel_fn = kernel_fn
        self.sigma_noise = sigma_noise
        self.use_train_set_for_test = use_train_set_for_test
        self.kernel_batch_sz = kernel_batch_sz
        self.keep_training_kernel_vals = keep_training_kernel_vals
        self.only_track_diag = only_track_diag  # covariance still returned as mxm, but only diagonal is correct
        self.indexs = None

        self.Xn = Xn
        self.yn = yn
        if self.keep_training_kernel_vals:
            self.K_nn = self.kernel_fn(x1=self.Xn)
            self.K_nn_diag = jnp.diag(self.K_nn)
        else:
            self.K_nn = None
            self.K_nn_diag = self.kernel_fn(x1=self.Xn, x2=None, get_diagonal_only=True)

        self.use_whole_train_for_test = (isinstance(self.use_train_set_for_test, bool) and 
                                         self.use_train_set_for_test)

        # kernel involving test data
        if self.keep_training_kernel_vals and self.use_whole_train_for_test:
            self.Xt = Xn
            self.K_tt = self.K_nn
            self.K_tt_diag = self.K_nn_diag
            self.K_nt = self.K_tt
        elif (self.keep_training_kernel_vals and 
              isinstance(self.use_train_set_for_test, jnp.ndarray) and 
              not self.use_whole_train_for_test):
            self.Xt = Xn[self.use_train_set_for_test, :]
            self.K_tt = self.K_nn[self.use_train_set_for_test, :][:, self.use_train_set_for_test]
            self.K_tt_diag = self.K_nn_diag[self.use_train_set_for_test]
            self.K_nt = self.K_nn[:, self.use_train_set_for_test]
        elif self.only_track_diag:
            assert Xt is not None
            self.Xt = Xt
            self.K_tt_diag = self.kernel_fn(x1=self.Xt, x2=None, get_diagonal_only=True)
            self.K_tt = None
            self.K_nt = self.kernel_fn(x1=self.Xn, x2=self.Xt)
        else:
            assert Xt is not None
            self.Xt = Xt
            self.K_tt = self.kernel_fn(x1=self.Xt, x2=None)
            self.K_tt_diag = jnp.diag(self.K_tt)
            self.K_nt = self.kernel_fn(x1=self.Xn, x2=self.Xt)
            
        if prior_mean_fn is None:
            self.yn_prior = jnp.zeros_like(self.yn)
            self.yt_prior = jnp.zeros(shape=(self.Xt.shape[0], self.yn_prior.shape[1]))
        else:
            self.yn_prior = prior_mean_fn(self.Xn)
            self.yt_prior = prior_mean_fn(self.Xt)

        self.update_train(jnp.arange(0))

    def update_train(self, indexs: jnp.ndarray):
        """
        Change the indices of training data
        and update precomputed kernel values
        """
        assert jnp.unique(indexs).shape[0] == indexs.shape[0]
        self.indexs = indexs
        self.yd = self.yn[self.indexs, :]
        self.yd_prior = self.yn_prior[self.indexs, :]

        if self.keep_training_kernel_vals:
            self.K_dd = self.K_nn[self.indexs, :][:, self.indexs]
            self.K_nd = self.K_nn[:, self.indexs]
            self.K_dt = self.K_nt[self.indexs, :]
        else:
            self.K_dd = self.kernel_fn(x1=self.Xn[self.indexs, :], x2=None)
            self.K_dt = self.K_nt[self.indexs, :]
            self.K_nd = self.kernel_fn(x1=self.Xn, x2=self.Xn[self.indexs, :])
        self.K_dd_sg_2_inv = jnp.linalg.inv(self.K_dd + self.sigma_noise ** 2 * jnp.eye(self.K_dd.shape[0]))
        self.test_pos_mean = self.yt_prior + self.K_dt.T @ self.K_dd_sg_2_inv @ (self.yd - self.yd_prior)
        if self.only_track_diag:
            self.test_pos_var = self.K_tt_diag - diag_matmul(self.K_dt.T, self.K_dd_sg_2_inv @ self.K_dt)
            self.test_pos_cov = None
        else:
            self.test_pos_cov = self.K_tt - self.K_dt.T @ self.K_dd_sg_2_inv @ self.K_dt
            self.test_pos_var = jnp.diag(self.test_pos_cov)

    def incrementally_update_train(self, additional_indexs: jnp.ndarray):
        new_idxs = jnp.block([self.indexs, additional_indexs])
        assert jnp.unique(new_idxs).shape[0] == self.indexs.shape[0] + additional_indexs.shape[0]
        if self.keep_training_kernel_vals:
            K_de = self.K_nn[self.indexs, :][:, additional_indexs]
            K_ee = self.K_nn[additional_indexs, :][:, additional_indexs]
            K_ne = self.K_nn[:, additional_indexs]
        else:
            K_de = self.kernel_fn(x1=self.Xn[self.indexs, :], x2=self.Xn[additional_indexs, :])
            K_ee = self.kernel_fn(x1=self.Xn[additional_indexs, :], x2=None)
            K_ne = self.kernel_fn(x1=self.Xn, x2=self.Xn[additional_indexs, :])
        K_et = self.K_nt[additional_indexs, :]
        self.K_dd = jnp.block([[self.K_dd, K_de], [K_de.T, K_ee]])
        self.K_dt = jnp.block([[self.K_dt], [K_et]])
        self.K_dd_sg_2_inv = symmetric_block_inverse(A_inv=self.K_dd_sg_2_inv, B=K_de,
                                                     D=K_ee + self.sigma_noise ** 2 * jnp.eye(K_ee.shape[0]))
        self.yd = jnp.block([[self.yd], [self.yn[additional_indexs, :]]])
        self.yd_prior = jnp.block([[self.yd_prior], [self.yn_prior[additional_indexs, :]]])
        
        self.test_pos_mean = self.yt_prior + self.K_dt.T @ self.K_dd_sg_2_inv @ (self.yd - self.yd_prior)
        if self.only_track_diag:
            self.test_pos_var = self.K_tt_diag - diag_matmul(self.K_dt.T, self.K_dd_sg_2_inv @ self.K_dt)
            self.test_pos_cov = None
        else:
            self.test_pos_cov = self.K_tt - self.K_dt.T @ self.K_dd_sg_2_inv @ self.K_dt
            self.test_pos_var = jnp.diag(self.test_pos_cov)
        
        self.K_nd = jnp.block([[self.K_nd, K_ne]])
        self.indexs = new_idxs

    def update_labels(self, new_yn: jnp.array, new_yt: jnp.array = None):
        self.yn = new_yn
        self.update_train(self.indexs)

    def get_train_posterior_mean(self):
        """
        Given the inducing point effective prior (as argument),
        compute the posterior of the test set conditioned on the prior
        exactly with the closed form formula
        """
        return self.yn_prior + self.K_nd @ self.K_dd_sg_2_inv @ (self.yd - self.yd_prior)

    def get_train_posterior_covariance(self):
        return self.K_nn - self.K_nd @ self.K_dd_sg_2_inv @ self.K_nd.T

    def get_test_posterior(self):
        """
        Get the mean and covariance of the test set conditioned on the selected points
        """
        assert not self.only_track_diag
        return self.test_pos_mean, self.test_pos_cov

    def get_test_posterior_diagonal(self):
        """
        Get the diagonal of posterior of the test set conditioned on the selected points
        """
        return self.test_pos_mean, self.test_pos_var

    def get_updated_test_posterior(self, idxs: jnp.ndarray):
        """
        Get the test posterior mean and covariance when trained on D + extra
        """
        if self.keep_training_kernel_vals:
            K_mm = self.K_nn[idxs, :][:, idxs]
        else:
            if len(idxs) == 1:
                K_mm = self.K_nn_diag[idxs].reshape(1, 1)
            else:
                K_mm = self.kernel_fn(self.Xn[idxs, :])
        K_mt = self.K_nt[idxs, :]
        K_md = self.K_nd[idxs, :]
        return _updated_test_distribution(K_dd_sg_2_inv=self.K_dd_sg_2_inv,
                                          K_mm_sg_2_I=K_mm + self.sigma_noise ** 2 * jnp.eye(K_mm.shape[0]),
                                          K_dt=self.K_dt,
                                          K_mt=K_mt,
                                          K_md=K_md,
                                          test_pos_mean=self.test_pos_mean,
                                          test_pos_cov=self.test_pos_cov,
                                          yd=self.yd - self.yd_prior,
                                          ym=self.yn[idxs, :] - self.yn_prior[idxs, :])
        
    def get_updated_test_posterior_diagonal(self, idxs: jnp.ndarray):
        """
        Get the test posterior mean and variance when trained on D + extra
        """
        if self.keep_training_kernel_vals:
            K_mm = self.K_nn[idxs, :][:, idxs]
        else:
            if len(idxs) == 1:
                K_mm = self.K_nn_diag[idxs].reshape(1, 1)
            else:
                K_mm = self.kernel_fn(self.Xn[idxs, :])
        K_mt = self.K_nt[idxs, :]
        K_md = self.K_nd[idxs, :]
        return _updated_test_distribution_diag(K_dd_sg_2_inv=self.K_dd_sg_2_inv,
                                               K_mm_sg_2_I=K_mm + self.sigma_noise ** 2 * jnp.eye(K_mm.shape[0]),
                                               K_dt=self.K_dt,
                                               K_mt=K_mt,
                                               K_md=K_md,
                                               test_pos_mean=self.test_pos_mean,
                                               test_pos_var=self.test_pos_var,
                                               yd=self.yd - self.yd_prior,
                                               ym=self.yn[idxs, :] - self.yn_prior[idxs, :])


@jax.jit
def _updated_test_distribution(K_dd_sg_2_inv, K_mm_sg_2_I, K_dt, K_mt, K_md, test_pos_mean, test_pos_cov, yd, ym):
    A_block_diff, B_block, D_block = symmetric_block_inverse_as_blocks_difference(A_inv=K_dd_sg_2_inv,
                                                                                  B=K_md.T,
                                                                                  D=K_mm_sg_2_I)

    P0 = K_dt.T @ A_block_diff @ K_dt
    P1 = K_dt.T @ B_block @ K_mt
    P2 = K_mt.T @ D_block @ K_mt
    additional_posterior = P0 + P1 + P1.T + P2

    M0 = K_dt.T @ A_block_diff @ yd
    M1 = K_dt.T @ B_block @ ym
    M2 = K_mt.T @ B_block.T @ yd
    M3 = K_mt.T @ D_block @ ym
    additional_mean = M0 + M1 + M2 + M3

    return test_pos_mean + additional_mean, test_pos_cov - additional_posterior


@jax.jit
def _updated_test_distribution_diag(K_dd_sg_2_inv, K_mm_sg_2_I, K_dt, K_mt, K_md, test_pos_mean, test_pos_var, yd, ym):
    A_block_diff, B_block, D_block = symmetric_block_inverse_as_blocks_difference(A_inv=K_dd_sg_2_inv,
                                                                                  B=K_md.T,
                                                                                  D=K_mm_sg_2_I)

    P0 = diag_matmul(K_dt.T, A_block_diff @ K_dt)
    P1 = diag_matmul(K_dt.T, B_block @ K_mt)
    P2 = diag_matmul(K_mt.T, D_block @ K_mt)
    additional_posterior = P0 + P1 + P1 + P2

    M0 = K_dt.T @ A_block_diff @ yd
    M1 = K_dt.T @ B_block @ ym
    M2 = K_mt.T @ B_block.T @ yd
    M3 = K_mt.T @ D_block @ ym
    additional_mean = M0 + M1 + M2 + M3

    return test_pos_mean + additional_mean, test_pos_var - additional_posterior

