import numpy as np
from signal_model.base_field_sampler import BaseFieldModelSampler, SamplerConfig, LineNormalization
from signal_model.spline_field.bspline import create_projection_matrix
import matplotlib.pyplot as plt


class BSplineSampler(BaseFieldModelSampler):
    def __init__(self, in_field, in_sensors, in_sampler_config: SamplerConfig):
        super().__init__(in_field, in_sensors, in_sampler_config)
        self.proj = self.create_projection_matrix()

    def dmu_dtheta(self, in_theta):
        return self.proj

    def dmu_dtheta_prior(self):
        return self.proj

    def create_projection_matrix(self, approximate=True, n_approximation_point=10000):
        H = create_projection_matrix(self.field,
                                     self.sensors,
                                     approximate=approximate,
                                     n_approximation_point=n_approximation_point)
        if self.is_line:
            length = self.length_line().reshape(-1, 1)

            if self.sampler_config.normalization_type == LineNormalization.L:
                return H
            elif self.sampler_config.normalization_type == LineNormalization.ONE:
                return H * length
            elif self.sampler_config.normalization_type == LineNormalization.SQRTL:
                return H * np.sqrt(length)
        return H


class BSplineMixerSampler:
    def __init__(self, in_field,
                 in_point,
                 in_link,
                 in_sampler_point: SamplerConfig,
                 in_sampler_link: SamplerConfig):
        self.sampler_point = BSplineSampler(in_field, in_point, in_sampler_point)
        self.sampler_link = BSplineSampler(in_field, in_link, in_sampler_link)

    def dmu_dtheta(self, in_theta):
        return np.vstack([self.sampler_point.dmu_dtheta(in_theta), self.sampler_link.dmu_dtheta(in_theta)])

    def dmu_dtheta_prior(self):
        return np.vstack([self.sampler_point.dmu_dtheta_prior(), self.sampler_link.dmu_dtheta_prior()])

    def inv_c_xx_bayesian(self, in_prior):
        return np.block([[self.sampler_point.inv_c_xx_bayesian(in_prior),
                          np.zeros((self.sampler_point.n_sensors, self.sampler_link.n_sensors))],
                         [np.zeros((self.sampler_link.n_sensors, self.sampler_point.n_sensors)),
                          self.sampler_link.inv_c_xx_bayesian(in_prior)]])

    def c_xx(self, in_theta):
        return np.block([[self.sampler_point.c_xx(in_theta),
                          np.zeros((self.sampler_point.n_sensors, self.sampler_link.n_sensors))],
                         [np.zeros((self.sampler_link.n_sensors, self.sampler_point.n_sensors)),
                          self.sampler_link.c_xx(in_theta)]])

    def inv_c_xx(self, in_theta):
        c_xx = self.c_xx(in_theta)
        return np.linalg.inv(c_xx)
