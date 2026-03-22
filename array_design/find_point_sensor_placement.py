import numpy as np

import bounds
from signal_model import SamplerConfig, LineNormalization
from signal_model.spline_field.bspline import BSplineRainField, create_projection_matrix
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import plot_sensors_map


class BSplinePointSensor(nn.Module):
    def __init__(self):
        super(BSplinePointSensor, self).__init__()

    pass


def find_point_sensors(n_p: int, d: int, link_sampler, in_rain_field: BSplineRainField,
                       projection2mse: bounds.Projection2MSE, in_sampler_point: SamplerConfig,
                       in_prior, scan_options=10,debug_plot = True):
    """
    Find the optimal placement of point sensors
    :return:
    """
    link_fim,_,_ = bounds.compute_bayesian_fisher_information_matrix(link_sampler,
                                                                 in_prior)  # Compute the Bayesian FIM for the Link sensors
    projection_matrix = in_rain_field.projection_matrix(d)  # Compute the projection matrix

    # Randomly select a point sensor
    x = np.random.rand(n_p) * in_rain_field.axis_size - in_rain_field.axis_size / 2
    y = np.random.rand(n_p) * in_rain_field.axis_size - in_rain_field.axis_size / 2
    sensors = np.vstack([x, y]).T
    sigma2 = in_sampler_point.alpha
    xx_yy = generate_scan_options(in_rain_field.axis_size, scan_options)
    mse_progress = []
    mse_step = []

    for iter in tqdm(range(7)):
        if debug_plot:
            P_tilde = create_projection_matrix(in_rain_field,
                                               sensors,
                                               in_sampler_point.normalization_type == LineNormalization.L)
            R = link_fim + P_tilde.T @ np.diag(np.ones(n_p) / sigma2) @ P_tilde
            bcrb = np.linalg.inv(R)
            bcrb_proj = projection_matrix.T @ bcrb @ projection_matrix
            field = bcrb_proj.diagonal().reshape([d, d])
            plot_bcrb_diagional_map(d, field, in_rain_field)
            plot_sensors_map(in_rain_field.axis_size, gauges=sensors, links=link_sampler.sensors, color="black")
            plt.tight_layout()
            plt.xlim([-in_rain_field.axis_size / 2, in_rain_field.axis_size / 2])
            plt.ylim([-in_rain_field.axis_size / 2, in_rain_field.axis_size / 2])
            plt.savefig(f"sensor_placement_{iter}.svg")
            plt.show()

        for sensor_index in range(n_p):
            if iter > 0:
                xx_yy_shift = generate_scan_options(in_rain_field.axis_size / 2 ** iter, scan_options)
                xx_yy = xx_yy_shift + sensors[sensor_index, :]

            sensors_filter = np.delete(sensors, sensor_index, axis=0)
            P_tilde = create_projection_matrix(in_rain_field,
                                               sensors_filter,
                                               in_sampler_point.normalization_type == LineNormalization.L)
            R_inv = np.linalg.inv(P_tilde.T @ np.diag(np.ones(n_p - 1) / sigma2) @ P_tilde + link_fim)
            del P_tilde
            mse_list = []
            for _xy in xx_yy:
                p_zero = create_projection_matrix(in_rain_field,
                                                  _xy.reshape(1, 2),
                                                  in_sampler_point.normalization_type == LineNormalization.L)
                bcrb = R_inv - (R_inv @ p_zero.T @ p_zero @ R_inv / sigma2) / (1 + p_zero @ R_inv @ p_zero.T / sigma2)
                bcrb_proj = projection_matrix.T @ bcrb @ projection_matrix
                mse = np.sqrt(np.trace(bcrb_proj) / d ** 2)
                mse_list.append(mse)
            current_mse = np.min(mse_list)
            if len(mse_progress) == 0 or (len(mse_progress) > 0 and current_mse < np.min(
                    mse_progress)):  # If the current MSE is less than the previous MSE then update the sensor position.
                mse_progress.append(np.min(mse_list))
                select_index = np.argmin(mse_list)
                sensors[sensor_index, :] = xx_yy[select_index, :]
        mse_step.append(mse_progress[-1])
        if len(mse_step) > 1 and mse_step[-1] == mse_step[-2]:
            break
        best_sensor = np.copy(sensors)

    return best_sensor


def plot_bcrb_diagional_map(d, field, in_rain_field):
    x = np.linspace(-in_rain_field.axis_size / 2, in_rain_field.axis_size / 2, d)
    xx, yy = np.meshgrid(x, x)
    plt.pcolormesh(xx, yy, field, cmap='OrRd', vmin=np.min(field), vmax=np.max(field))
    plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    plt.colorbar()


def generate_scan_options(axis_size, scan_options):
    xy = np.linspace(- axis_size / 2, axis_size / 2, scan_options)
    xx_yy_grid = np.meshgrid(xy, xy)
    xx_yy = np.vstack([xx_yy_grid[0].flatten(), xx_yy_grid[1].flatten()]).T
    return xx_yy
