from signal_model.base_field_sampler import BaseFieldModelSampler
from signal_model.prior.base_prior import BasePrior
import numpy as np


def compute_bayesian_fisher_information_matrix(in_sampler: BaseFieldModelSampler, in_prior: BasePrior, scale=1):
    h = in_sampler.dmu_dtheta_prior()
    inv_c_xx = in_sampler.inv_c_xx_bayesian(in_prior)
    efim = h.T @ inv_c_xx @ h * scale ** 2
    pfim = in_prior.prior_fim()
    bfim = efim + pfim
    return np.linalg.inv(bfim), bfim, efim, pfim
