import numpy as np
from enum import Enum


class LineNormalization(Enum):
    L = 0
    ONE = 1
    SQRTL = 2


class SamplerConfig(object):
    def __init__(self, alpha, beta,
                 normalization_type: LineNormalization = LineNormalization.L):
        """
        Sensor Sampling Noise Configuration
        :param alpha:
        :param beta:
        :param normalization_type:
        """
        self.alpha = alpha
        self.beta = beta
        self.normalization_type = normalization_type


class BaseFieldModelSampler(object):
    def __init__(self, in_field, in_sensors, in_sampler_config: SamplerConfig):
        self.field = in_field
        self.sensors = in_sensors
        self.sampler_config = in_sampler_config

    def length_line(self):
        if self.is_line:
            return np.sqrt(
                (self.sensors[:, 0] - self.sensors[:, 1]) ** 2 + (self.sensors[:, 2] - self.sensors[:, 3]) ** 2)
        raise Exception("Length can be compute only for line sensors")

    @property
    def is_line(self) -> bool:
        return self.sensors.shape[-1] == 4

    @property
    def n_sensors(self):
        return self.sensors.shape[0]

    def dmu_dtheta(self, in_theta):
        raise NotImplemented

    def dmu_dtheta_prior(self):
        pass

    def inv_c_xx_bayesian(self, in_prior):
        return self.inv_c_xx(in_prior)


    def c_xx(self, in_theta):
        if self.is_line:
            l = self.length_line()
            sigma2 = self.sampler_config.alpha + self.sampler_config.beta * l
            return np.diag(sigma2)
        else:
            sigma2 = self.sampler_config.alpha
            return np.diag(np.ones(self.n_sensors) * sigma2)

    def inv_c_xx(self, in_theta):
        c_xx = self.c_xx(in_theta)
        return np.linalg.inv(c_xx)

