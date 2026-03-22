import numpy as np


class GammaPrior:
    def __init__(self, n_param, alpha, beta):
        self.beta = None
        self.alpha = None
        self.scale = None
        self.n_param = n_param

        self.update_params(alpha, beta)

    def update_params(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        beta_pot = (beta ** 2)
        self.scale = beta_pot / (alpha - 2)

    def prior_fim(self):
        if self.scale is None:
            raise Exception("Please set scale")
        c_xx = np.diag(self.scale * np.ones(self.n_param))
        return c_xx
