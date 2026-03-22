import numpy as np
import bounds
from config import BoundTypes


def compute_all_bounds(bound_type, bspline_field, parameter, projection2mse,
                       gauge_sampler, link_sampler, link_misspecifed_sampler, mean_length, resample=False):
    if bound_type == BoundTypes.CRB:
        theta = parameter.get_theta(bspline_field.n_parameters(), resample=resample)
        gauge_crb = bounds.compute_fisher_information_matrix(gauge_sampler, theta)
        link_crb = bounds.compute_fisher_information_matrix(link_sampler, theta)
        link_mcrb = bounds.compute_mcrb(link_sampler, link_misspecifed_sampler, theta, parameter.get_covariance(),
                                        parameter.get_mean())
    else:
        parameter.set_n_parameter(bspline_field.n_parameters())
        prior = parameter.get_prior(bspline_field.n_parameters())
        gauge_crb, _, _, _ = bounds.compute_bayesian_fisher_information_matrix(gauge_sampler,prior)  # The scale is used to adjust the noise level to match the link noise level.
        link_crb, _, _, _ = bounds.compute_bayesian_fisher_information_matrix(link_sampler, prior)
        link_mcrb = bounds.compute_bmcrb(link_sampler, link_misspecifed_sampler,prior, parameter.get_covariance(),
                                         parameter.get_mean())
    mse_gauge, crb_gauge = projection2mse(gauge_crb)
    mse_link, crb_link = projection2mse(link_crb)
    if link_mcrb is not None:
        mse_mcrb, crb_link = projection2mse(link_mcrb)
    else:
        mse_mcrb = np.nan
    return mse_gauge, mse_link, mse_mcrb
