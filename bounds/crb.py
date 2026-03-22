import numpy as np
from signal_model.base_field_sampler import BaseFieldModelSampler


def compute_fisher_information_matrix(in_sampler: BaseFieldModelSampler, in_theta, scale=1.0):
    h = in_sampler.dmu_dtheta(in_theta)
    inv_c_xx = in_sampler.inv_c_xx(in_theta)
    fim = h.T @ inv_c_xx @ h * scale ** 2
    return np.linalg.inv(fim)
