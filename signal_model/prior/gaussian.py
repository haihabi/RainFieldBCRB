import numpy as np
from signal_model.prior.base_prior import BasePrior


class GaussianPrior(BasePrior):
    def __init__(self, n_param, sigma, corrlation_factor=0.0):
        self.sigma = sigma
        self.n_param = n_param
        self.corrlation_factor = corrlation_factor

    def prior_fim(self):
        c_xx = np.diag((self.sigma ** 2) * np.ones(self.n_param)) + np.ones(
            (self.n_param, self.n_param)) * self.corrlation_factor

        return np.linalg.inv(c_xx)
