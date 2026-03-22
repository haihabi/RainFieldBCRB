import numpy as np



def compute_mcrb(in_true_sampler, in_assumed_sampler, in_theta, covariance, mean):
    h_assumed = in_assumed_sampler.dmu_dtheta(in_theta)
    inv_c_xx_assumed = in_assumed_sampler.inv_c_xx(in_theta)
    c_xx_true = in_true_sampler.c_xx(in_theta)
    h_true = in_true_sampler.dmu_dtheta(in_theta)

    a_part = h_assumed.T @ inv_c_xx_assumed
    a_matrix = a_part @ h_assumed
    b_matrix = a_part @ c_xx_true @ a_part.T

    a_matrix_inv = np.linalg.inv(a_matrix)
    mcrb = a_matrix_inv @ b_matrix @ a_matrix_inv

    P = a_matrix_inv @ a_part @ h_true - np.eye(a_matrix_inv.shape[0])  # Bias matrix
    mean_vec = mean.reshape(-1, 1) @ mean.reshape(1, -1)  # Reshape mean to a column vector
    rrt_cov = P @ (covariance + mean_vec) @ P.T  # Bias covariance

    return mcrb + rrt_cov


def compute_bmcrb(in_true_sampler, in_assumed_sampler,in_prior, covariance, mean):
    """
    Compute the MCRB for the given true and assumed samplers.

    Parameters:
        in_true_sampler: The true field model sampler.
        in_assumed_sampler: The assumed field model sampler.
        in_theta: The parameter vector.
        covariance: The covariance matrix of the parameter.
        mean: The mean vector of the parameter.

    Returns:
        mcrb: The computed MCRB matrix.
    """
    h_assumed = in_assumed_sampler.dmu_dtheta(None)
    inv_c_xx_assumed = in_assumed_sampler.inv_c_xx(None)
    c_xx_true = in_true_sampler.c_xx(None)
    h_true = in_true_sampler.dmu_dtheta(None)
    #
    a_part = h_assumed.T @ inv_c_xx_assumed
    a_matrix = a_part @ h_assumed

    h_f = a_matrix + np.linalg.inv(covariance)
    h_f_inv = np.linalg.inv(h_f)
    P = h_f_inv @ a_part @ h_true
    J = np.linalg.inv(h_true.T @ np.linalg.inv(c_xx_true) @ h_true + in_prior.prior_fim())
    mbcrb = P @ J @ P.T  # MCRB matrix
    # Compute the bias vector
    delta_h =h_true-h_assumed
    mean_mat = mean.reshape(-1, 1) @ mean.reshape(1, -1)  # Reshape mean to a column vector

    P1 = a_part @ delta_h
    P2 = P1 - np.linalg.inv(covariance)
    T1 = P1 @ mean_mat @ P1.T  # Bias covariance
    T2 = P2 @ covariance @ P2.T  # Bias covariance
    R = h_f_inv @ (T1 + T2) @ h_f_inv.T  # Bias covariance
    return mbcrb+R
