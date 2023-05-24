import jax.numpy as jnp


class GP:

    def __init__(self):
        pass

    def update_train(self, indexs: jnp.ndarray):
        """
        Change the indices of training data
        and update precomputed kernel values
        """
        raise NotImplementedError

    # def incrementally_update_train(self, additional_indexs: jnp.ndarray):
    #     raise NotImplementedError

    def update_labels(self, new_yn: jnp.array, new_yt: jnp.array = None):
        raise NotImplementedError

    def get_train_posterior_mean(self):
        """
        Given the inducing point effective prior (as argument),
        compute the posterior of the test set conditioned on the prior
        exactly with the closed form formula
        """
        raise NotImplementedError

    def get_train_posterior_covariance(self):
        raise NotImplementedError

    def get_test_posterior(self):
        """
        Get the mean and covariance of the test set conditioned on the selected points
        """
        raise NotImplementedError

    def get_test_posterior_diagonal(self):
        """
        Get the diagonal of posterior of the test set conditioned on the selected points
        """
        mean, cov = self.get_test_posterior()
        return mean, jnp.diag(cov)

    def get_updated_test_posterior(self, idxs: jnp.ndarray):
        """
        Get the test posterior mean and covariance when trained on D + extra
        """
        raise NotImplementedError
    
    def get_updated_test_posterior_diagonal(self, idxs: jnp.ndarray):
        """
        Get the test posterior mean and covariance when trained on D + extra
        """
        mean, cov = self.get_updated_test_posterior(idxs=idxs)
        return mean, jnp.diag(cov)
