import numpy as np

import signal_model


class Parameter:
    def __init__(self, n_parameter, peak_rain_rate, variance=2.3):
        self.n_parameter = n_parameter
        self.peak_rain_rate = peak_rain_rate
        self.theta = None

        self.beta = peak_rain_rate/variance
        self.alpha = self.beta * peak_rain_rate

    def set_n_parameter(self, n_parameter):
        if n_parameter != self.n_parameter:
            self.n_parameter = n_parameter
            self.theta = None

    def get_theta(self, n_parameter=None, resample=False):
        if n_parameter != self.n_parameter or self.theta is None or resample:
            self.n_parameter = n_parameter if n_parameter is not None else self.n_parameter
            theta_base = np.random.rand(n_parameter)
            theta_base /= np.linalg.norm(theta_base)
            scale = self.peak_rain_rate / np.max(theta_base)
            self.theta = theta_base * scale

        return self.theta

    def get_covariance(self):
        diag = self.alpha / (self.beta ** 2)
        return np.eye(self.n_parameter) * diag

    def get_mean(self):
        return self.peak_rain_rate * np.ones(self.n_parameter)

    def get_prior(self, n_parameter):
        return signal_model.prior.GammaPrior(n_param=n_parameter, alpha=self.alpha, beta=self.beta)
