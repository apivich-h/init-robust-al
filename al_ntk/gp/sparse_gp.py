from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
from functools import partial

from sklearn.cluster import KMeans
import jax
import jax.numpy as jnp

from al_ntk.dataset import Dataset
from al_ntk.gp.gp import GP
from al_ntk.utils.linalg_jax import symmetric_block_inverse, diag_matmul
from al_ntk.utils.entropy_helper import max_entropy_selector_from_fn
from al_ntk.utils.kernels_helper import compute_kernel_in_batches


class SparseGP(ABC, GP):

    # typing for kernel_fn is probably wrong...
    def __init__(self, kernel_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray], Optional[bool]], jnp.ndarray],
                 Xn: jnp.array, yn: jnp.array, Xt: jnp.array = None, yt: jnp.array = None, sigma_noise: float = 1e-2,
                 keep_training_kernel_vals: bool = False, inducing_pts: Union[jnp.ndarray, str] = 'kmeans',
                 use_train_set_for_test: Union[bool, jnp.ndarray] = False, only_track_diag: bool = True,
                 m: int = 50, inducing_pt_ent_bound: float = -10.):
        super().__init__()
        self.kernel_fn = kernel_fn
        self.sigma_noise = sigma_noise
        self.use_train_set_for_test = use_train_set_for_test
        self.keep_training_kernel_vals = keep_training_kernel_vals
        self.inducing_pt_ent_bound = inducing_pt_ent_bound
        self.only_track_diag = only_track_diag

        self.Xn = Xn
        self.yn = yn
        
        # kernels involving training data
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
            self.yt = yn
            self.K_tt = self.K_nn
            self.K_tt_diag = self.K_nn_diag
        elif self.keep_training_kernel_vals and self.use_train_set_for_test and not self.use_whole_train_for_test:
            self.Xt = Xn[self.use_train_set_for_test, :]
            self.yt = yn[self.use_train_set_for_test, :] if (yn is not None) else None
            self.K_tt = self.K_nn[self.use_train_set_for_test, :][:, self.use_train_set_for_test]
            self.K_tt_diag = self.K_nn_diag[self.use_train_set_for_test]
        elif self.only_track_diag:
            assert Xt is not None
            self.Xt = Xt
            self.yt = yt
            self.K_tt_diag = self.kernel_fn(x1=self.Xt, x2=None, get_diagonal_only=True)
            self.K_tt = None
        else:
            assert Xt is not None
            self.Xt = Xt
            self.yt = yt
            self.K_tt = self.kernel_fn(x1=self.Xt, x2=None)
            self.K_tt_diag = jnp.diag(self.K_tt)

        self.m = m
        if isinstance(inducing_pts, jnp.ndarray):
            self.Xu = inducing_pts
        else:
            self.Xu: Optional[jnp.ndarray] = self.get_inducing_points(method=inducing_pts)
        self.m = self.Xu.shape[0]

        # kernel involving inducing points
        self.K_uu = self.kernel_fn(x1=self.Xu, x2=None)
        self.K_uu_inv = jnp.linalg.inv(self.K_uu)
        self.K_uu_noise_inv = jnp.linalg.inv(self.K_uu + self.sigma_noise ** 2 * jnp.eye(self.m))

        # kernel involving selected training data
        self.indexs = None
        self.yd = None
        self.K_du = None
        self.K_ud = None
        self.F_batch_selected = None
        self.F_batch_selected_sum = None
        self.G_batch_selected = None
        self.G_batch_selected_sum = None

        # kernel involving test data
        self.K_nu = self.kernel_fn(x1=self.Xn, x2=self.Xu)
        self.K_tu = self.kernel_fn(x1=self.Xt, x2=self.Xu)
        self.K_ut = self.K_tu.transpose()

    def get_inducing_points(self, method: str = 'kmeans'):
        if method == 'kmeans':
            data_shape = self.Xt.shape[1:]
            Xt_flatten = self.Xt.reshape((self.Xt.shape[0], -1))
            kmeans = KMeans(n_clusters=self.m).fit(Xt_flatten)
            inducing_pts = kmeans.cluster_centers_.reshape((self.m, *data_shape))
        elif method == 'entropy':
            idxs = max_entropy_selector_from_fn(
                kernel_fn=self.kernel_fn,
                xs=self.Xt,
                m=self.m,
                entropy_lb=self.inducing_pt_ent_bound
            )
            inducing_pts = self.Xt[idxs, :]
        elif method == 'mixed':
            idxs = max_entropy_selector_from_fn(
                kernel_fn=self.kernel_fn,
                xs=self.Xt,
                m=self.m // 2,
                entropy_lb=self.inducing_pt_ent_bound
            )
            remaining = list(set(range(self.Xt.shape[0])).difference(idxs))
            inducing_pts = self.Xt[idxs, :]
            kmeans = KMeans(n_clusters=self.m - len(idxs)).fit(self.Xt[remaining, :])
            inducing_pts = jnp.concatenate([inducing_pts, kmeans.cluster_centers_])
        else:
            raise ValueError('Invalid inducing_pts parameter.')
        return jnp.array(inducing_pts)

    def update_train(self, indexs: jnp.ndarray):
        """
        Change the indices of training data
        and update precomputed kernel values
        """
        self.indexs = indexs
        self.yd = self.yn[indexs, :]
        self.K_du = self.K_nu[self.indexs, :]
        self.K_ud = self.K_du.transpose()
        self.F_batch_selected, self.G_batch_selected = self.get_F_and_G_batch(idxs=self.indexs)
        self.F_batch_selected_sum = jnp.sum(self.F_batch_selected, axis=0)
        self.G_batch_selected_sum = jnp.sum(self.G_batch_selected, axis=0)
        self._internal_update_train(indexs)

        sum_F = self.F_batch_selected_sum + self.get_F_prime()
        sum_G = self.G_batch_selected_sum + self.get_G_prime()
        self.qu_theta1, self.qu_theta2 = sum_G, -0.5 * sum_F
        self.qu_mean, self.qu_cov = self.get_mean_and_cov(theta_1=self.qu_theta1, theta_2=self.qu_theta2)

        self.train_pos_mean = None
        self.train_pos_cov = None
        self.train_pos_var = None
        
        self.test_pos_mean = None
        self.test_pos_cov = None
        self.test_pos_var = None

    @abstractmethod
    def _internal_update_train(self, indexs: jnp.ndarray):
        raise NotImplementedError

    def incrementally_update_train(self, additional_indexs: jnp.ndarray):
        new_idxs = jnp.block([self.indexs, additional_indexs])
        assert jnp.unique(new_idxs).shape[0] == self.indexs.shape[0] + additional_indexs.shape[0]
        self.update_train(new_idxs)

    @abstractmethod
    def _internal_incrementally_update_train(self, additional_indexs: jnp.ndarray):
        raise NotImplementedError

    def update_labels(self, new_yn: jnp.array, new_yt: jnp.array = None):
        self.yn = new_yn
        if self.keep_training_kernel_vals and self.use_whole_train_for_test:
            self.yt = new_yn
        elif self.keep_training_kernel_vals and not self.use_whole_train_for_test:
            self.yt = new_yn[self.use_train_set_for_test, :]
        else:
            self.yt = new_yt
        self._internal_update_labels()
        self.update_train(self.indexs)

    @abstractmethod
    def _internal_update_labels(self):
        raise NotImplementedError

    def get_train_posterior_mean(self) -> jnp.ndarray:
        if self.train_pos_mean is None:
            self.train_pos_mean = _test_posterior_mean(
                K_uu_noise_inv=self.K_uu_inv,
                K_ut=self.K_nu.T,
                K_tu=self.K_nu,
                qu_mean=self.qu_mean,
                qu_cov=self.qu_cov
            )
        return self.train_pos_mean

    def get_train_posterior_covariance(self):
        assert not self.only_track_diag
        if self.train_pos_cov is None:
            self.train_pos_cov = _test_posterior_cov(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_nu.T,
                K_tu=self.K_nu,
                K_tt=self.K_nn,
                qu_mean=self.qu_mean,
                qu_cov=self.qu_cov
            )
            self.train_pos_var = jnp.diag(self.train_pos_cov)
        return self.train_pos_cov
    
    def get_train_posterior_variance(self):
        if self.train_pos_var is None:
            self.train_pos_var = _test_posterior_var(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_nu.T,
                K_tu=self.K_nu,
                K_tt_diag=self.K_nn_diag,
                qu_mean=self.qu_mean,
                qu_cov=self.qu_cov
            )
        return self.train_pos_var

    def get_test_posterior(self):
        """
        Given the inducing point effective prior (as argument),
        compute the posterior of the test set conditioned on the prior
        exactly with the closed form formula
        """
        assert not self.only_track_diag
        if self.test_pos_mean is None or self.test_pos_cov is None:
            self.test_pos_mean, self.test_pos_cov = _test_posterior_mean_cov(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt=self.K_tt,
                qu_mean=self.qu_mean,
                qu_cov=self.qu_cov
            )
        return self.test_pos_mean, self.test_pos_cov
        
    def get_test_posterior_diagonal(self):
        """
        Given the inducing point effective prior (as argument),
        compute the posterior of the test set conditioned on the prior
        exactly with the closed form formula
        """
        
        if self.test_pos_mean is None or self.test_pos_var is None:
        
            if self.only_track_diag:
                self.test_pos_mean, self.test_pos_var = _test_posterior_mean_var(
                    K_uu_noise_inv=self.K_uu_noise_inv,
                    K_ut=self.K_ut,
                    K_tu=self.K_tu,
                    K_tt_diag=self.K_tt_diag,
                    qu_mean=self.qu_mean,
                    qu_cov=self.qu_cov
                )
        
            else:
                self.test_pos_mean, self.test_pos_cov = _test_posterior_mean_cov(
                    K_uu_noise_inv=self.K_uu_noise_inv,
                    K_ut=self.K_ut,
                    K_tu=self.K_tu,
                    K_tt=self.K_tt,
                    qu_mean=self.qu_mean,
                    qu_cov=self.qu_cov
                )
                self.test_pos_var = jnp.diag(self.test_pos_cov)
        
        return self.test_pos_mean, self.test_pos_var

    @abstractmethod
    def get_dL_extra_separate(self, idxs: jnp.ndarray = None):
        """
        Derivative of (extra training loss when extra data added to training) wrt inducing covariance
        """
        raise NotImplementedError

    @abstractmethod
    def get_dL_extra(self, idxs: jnp.ndarray = None):
        """
        The difference in gradient of variational loss when new point at index idx is added
        """
        raise NotImplementedError

    @abstractmethod
    def get_dL_extra_separate_nat_param(self, idxs: jnp.ndarray = None):
        """
        Derivative of (extra training loss when extra data added to training) wrt inducing covariance
        """
        raise NotImplementedError

    @abstractmethod
    def get_dL_extra_nat_param(self, idxs: jnp.ndarray = None):
        """
        The difference in gradient of variational loss when new point at index idx is added
        """
        raise NotImplementedError

    def get_updated_inducing_distribution(self, idxs: jnp.ndarray):
        """
        Given inducing posterior covariance when trained on D,
        get the inducing posterior covariance when trained on D + extra
        """
        F_extra, G_extra = self.get_F_and_G_batch(idxs)
        t1, t2 = self.qu_theta1 + jnp.sum(G_extra, axis=0), self.qu_theta2 - 0.5 * jnp.sum(F_extra, axis=0)
        return self.get_mean_and_cov(theta_1=t1, theta_2=t2)

    def get_updated_test_posterior(self, idxs: jnp.ndarray):
        """
        Given inducing posterior covariance when trained on D,
        get the test posterior covariance when trained on D + extra
        """
        updated_qu_mean, updated_qu_cov = self.get_updated_inducing_distribution(idxs=idxs)
        if self.only_track_diag:
            return _test_posterior_mean_var(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt_diag=self.K_tt_diag,
                qu_mean=updated_qu_mean,
                qu_cov=updated_qu_cov
            )
        else:
            return _test_posterior_mean_cov(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt=self.K_tt,
                qu_mean=updated_qu_mean,
                qu_cov=updated_qu_cov
            )

    def get_grad_wrt_nat_param(self, func):
        if self.only_track_diag:
            return _grad_wrt_nat_param_with_var(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt_diag=self.K_tt_diag,
                theta_1=self.qu_theta1,
                theta_2=self.qu_theta2,
                loss_fn=func,
                yt=self.yt
            )
        else:
            return _grad_wrt_nat_param_with_cov(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt=self.K_tt,
                theta_1=self.qu_theta1,
                theta_2=self.qu_theta2,
                loss_fn=func,
                yt=self.yt
            )

    def get_jacobian_wrt_nat_param(self, func):
        if self.only_track_diag:
            return _jacobian_wrt_nat_param_with_var(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt_diag=self.K_tt_diag,
                theta_1=self.qu_theta1,
                theta_2=self.qu_theta2,
                func=func,
                yt=self.yt
            )
        else:
            return _jacobian_wrt_nat_param_with_cov(
                K_uu_noise_inv=self.K_uu_noise_inv,
                K_ut=self.K_ut,
                K_tu=self.K_tu,
                K_tt=self.K_tt,
                theta_1=self.qu_theta1,
                theta_2=self.qu_theta2,
                func=func,
                yt=self.yt
            )

    @abstractmethod
    def get_F_prime(self):
        raise NotImplementedError

    @abstractmethod
    def get_G_prime(self):
        raise NotImplementedError

    @abstractmethod
    def get_F_and_G_batch(self, idxs: jnp.ndarray = None):
        raise NotImplementedError

    def get_natural_params(self, qu_mean: jnp.ndarray, qu_cov: jnp.ndarray):
        return _natural_params(qu_mean=qu_mean, qu_cov=qu_cov)

    def get_mean_and_cov(self, theta_1: jnp.ndarray, theta_2: jnp.ndarray):
        return _mean_and_cov(t1=theta_1, t2=theta_2)

    def get_inducing_posterior_natural_param(self):
        """
        Update the inducing posterior using natural params
        """
        return self.qu_theta1, self.qu_theta2

    def get_inducing_posterior(self):
        """
        Update the inducing posterior through gradient descent (not sgd yet)
        given mean and covariance
        (done by converting it to natural param, do gd, then convert back)
        """
        return self.qu_mean, self.qu_cov

    def get_change_fn_given_added_element(self, func):
        dloss_dt1, dloss_dt2 = self.get_grad_wrt_nat_param(func=func)
        extra_dL_t1, extra_dL_t2 = self.get_dL_extra_separate_nat_param()
        scores = (jnp.sum(dloss_dt1[None, :, :] * extra_dL_t1, axis=(1, 2)) +
                  jnp.sum(dloss_dt2[None, :, :] * extra_dL_t2, axis=(1, 2)))
        return scores

    def get_change_multivar_fn_given_added_element(self, func):
        dloss_dt1, dloss_dt2 = self.get_jacobian_wrt_nat_param(func=func)
        extra_dL_t1, extra_dL_t2 = self.get_dL_extra_separate_nat_param()
        scores = (jnp.sum(dloss_dt1[:, None] * extra_dL_t1[None, :], axis=(2, 3)) +
                  jnp.sum(dloss_dt2[:, None] * extra_dL_t2[None, :], axis=(2, 3)))
        return scores


@jax.jit
def _test_posterior_mean(K_uu_noise_inv, K_ut, K_tu, qu_mean, qu_cov):
    return K_tu @ K_uu_noise_inv @ qu_mean


@jax.jit
def _test_posterior_cov(K_uu_noise_inv, K_ut, K_tu, K_tt, qu_mean, qu_cov):
    return K_tt + K_tu @ ((K_uu_noise_inv @ qu_cov @ K_uu_noise_inv) - K_uu_noise_inv) @ K_ut


@jax.jit
def _test_posterior_var(K_uu_noise_inv, K_ut, K_tu, K_tt_diag, qu_mean, qu_cov):
    return K_tt_diag + diag_matmul(K_tu, ((K_uu_noise_inv @ qu_cov @ K_uu_noise_inv) - K_uu_noise_inv) @ K_ut)


@jax.jit
def _test_posterior_mean_cov(K_uu_noise_inv, K_ut, K_tu, K_tt, qu_mean, qu_cov):
    mean = _test_posterior_mean(K_uu_noise_inv, K_ut, K_tu, qu_mean, qu_cov)
    cov = _test_posterior_cov(K_uu_noise_inv, K_ut, K_tu, K_tt, qu_mean, qu_cov)
    return mean, cov


@jax.jit
def _test_posterior_mean_var(K_uu_noise_inv, K_ut, K_tu, K_tt_diag, qu_mean, qu_cov):
    mean = _test_posterior_mean(K_uu_noise_inv, K_ut, K_tu, qu_mean, qu_cov)
    var = _test_posterior_var(K_uu_noise_inv, K_ut, K_tu, K_tt_diag, qu_mean, qu_cov)
    return mean, var


@partial(jax.jit, static_argnames=['m', 'loss_fn'])
def _test_posterior_derivative_with_cov(K_uu_noise_inv, K_ut, K_tu, K_tt, qu_mean, qu_cov, m, loss_fn):
    def fn(mean, cov_flatten):
        t_mean, t_cov = _test_posterior_mean_cov(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt=K_tt,
            qu_mean=mean,
            qu_cov=cov_flatten.reshape(m, m)
        )
        return loss_fn(mean=t_mean, cov=t_cov)

    d_mean, d_cov = jax.grad(fn, argnums=2)(qu_mean, qu_cov.flatten())
    return d_mean, d_cov.reshape(m, m)


@partial(jax.jit, static_argnames=['m', 'loss_fn'])
def _test_posterior_derivative_with_var(K_uu_noise_inv, K_ut, K_tu, K_tt_diag, qu_mean, qu_cov, m, loss_fn):
    def fn(mean, cov_flatten):
        t_mean, t_var = _test_posterior_mean_var(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt_diag=K_tt_diag,
            qu_mean=mean,
            qu_cov=cov_flatten.reshape(m, m)
        )
        return loss_fn(mean=t_mean, var=t_var)

    d_mean, d_cov = jax.grad(fn, argnums=2)(qu_mean, qu_cov.flatten())
    return d_mean, d_cov.reshape(m, m)


@partial(jax.jit, static_argnames=['loss_fn'])
def _grad_wrt_nat_param_with_cov(yt, K_uu_noise_inv, K_ut, K_tu, K_tt, theta_1, theta_2, loss_fn):
    def fn(theta):
        qu_mean, qu_cov = _mean_and_cov(t1=theta[1], t2=theta[2])
        test_mean, test_cov = _test_posterior_mean_cov(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt=K_tt,
            qu_mean=qu_mean,
            qu_cov=qu_cov
        )
        return loss_fn(ys=yt, mean=test_mean, cov=test_cov)

    grad = jax.grad(fn)({1: theta_1, 2: theta_2})
    return grad[1], grad[2]


@partial(jax.jit, static_argnames=['loss_fn'])
def _grad_wrt_nat_param_with_var(yt, K_uu_noise_inv, K_ut, K_tu, K_tt_diag, theta_1, theta_2, loss_fn):
    def fn(theta):
        qu_mean, qu_cov = _mean_and_cov(t1=theta[1], t2=theta[2])
        test_mean, test_var = _test_posterior_mean_var(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt_diag=K_tt_diag,
            qu_mean=qu_mean,
            qu_cov=qu_cov
        )
        return loss_fn(ys=yt, mean=test_mean, var=test_var)

    grad = jax.grad(fn)({1: theta_1, 2: theta_2})
    return grad[1], grad[2]


@partial(jax.jit, static_argnames=['func'])
def _jacobian_wrt_nat_param_with_cov(yt, K_uu_noise_inv, K_ut, K_tu, K_tt, theta_1, theta_2, func):
    def fn(theta):
        qu_mean, qu_cov = _mean_and_cov(t1=theta[1], t2=theta[2])
        test_mean, test_cov = _test_posterior_mean_cov(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt=K_tt,
            qu_mean=qu_mean,
            qu_cov=qu_cov
        )
        return func(ys=yt, mean=test_mean, cov=test_cov)

    jac = jax.jacrev(fn)({1: theta_1, 2: theta_2})
    return jac[1], jac[2]


@partial(jax.jit, static_argnames=['func'])
def _jacobian_wrt_nat_param_with_var(yt, K_uu_noise_inv, K_ut, K_tu, K_tt_diag, theta_1, theta_2, func):
    def fn(theta):
        qu_mean, qu_cov = _mean_and_cov(t1=theta[1], t2=theta[2])
        test_mean, test_var = _test_posterior_mean_var(
            K_uu_noise_inv=K_uu_noise_inv,
            K_ut=K_ut,
            K_tu=K_tu,
            K_tt_diag=K_tt_diag,
            qu_mean=qu_mean,
            qu_cov=qu_cov
        )
        return func(ys=yt, mean=test_mean, var=test_var)

    jac = jax.jacrev(fn)({1: theta_1, 2: theta_2})
    return jac[1], jac[2]


@jax.jit
def _natural_params(qu_mean: jnp.ndarray, qu_cov: jnp.ndarray):
    qu_cov_inv = jnp.linalg.inv(qu_cov)
    return qu_cov_inv @ qu_mean, -0.5 * qu_cov_inv


@jax.jit
def _mean_and_cov(t1: jnp.ndarray, t2: jnp.ndarray):
    qu_cov = -0.5 * jnp.linalg.inv(t2)
    qu_mean = qu_cov @ t1
    return qu_mean, qu_cov
