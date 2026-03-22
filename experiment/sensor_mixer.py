import numpy as np
from tqdm import tqdm

import bounds
import signal_model
from array_design.find_point_sensor_placement import find_point_sensors, plot_bcrb_diagional_map
from config import BoundTypes
from signal_model import BSplineRainFieldGenerator
from utils import plot_sensors_map
from matplotlib import pyplot as plt


def run_sensor_mixer(links, n_gauge_sensors, bspline_generator: BSplineRainFieldGenerator, n_projection,
                     bound_type, sampler_cfg_gauge, sampler_cfg_link, parameter, n_mc=10, order=3, debug_plot=False):
    slg_gauge = signal_model.SensorGenerator(
        map_shape=(bspline_generator.axis_size / 2, bspline_generator.axis_size / 2), n_sensors=n_gauge_sensors)

    # for order in tqdm(order_list):
    bspline_field = bspline_generator.generate_filed(order)
    projection2mse = bounds.Projection2MSE(bspline_field, n_projection, bounds.ProjectionType.FieldMSE)
    link_sampler = signal_model.generate_sampler(bspline_field, links, sampler_cfg_link)
    # Grid mixer
    delta = bspline_generator.axis_size / (np.ceil(np.sqrt(n_gauge_sensors)) + 1)
    gauges = slg_gauge.generate_gauge_position(is_center_uniform=True, shift=delta / 2)
    mixer_sampler = signal_model.generate_mixed_sampler(bspline_field, gauges, links, sampler_cfg_gauge,
                                                        sampler_cfg_link)

    mse_grid, bcrb = compute_mse(bound_type, mixer_sampler, parameter, projection2mse)
    if debug_plot:
        field = bcrb.diagonal().reshape([n_projection, n_projection])
        plot_bcrb_diagional_map(n_projection, field, bspline_field)
        plot_sensors_map(bspline_field.axis_size, gauges=gauges, links=link_sampler.sensors, color="black")
        plt.tight_layout()
        plt.xlim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.ylim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.savefig(f"analysis/results/sensor_grid.svg")
        plt.show()



        _mse_grid, _bcrb = compute_mse(bound_type, link_sampler, parameter, projection2mse)

        field = _bcrb.diagonal().reshape([n_projection, n_projection])
        plot_bcrb_diagional_map(n_projection, field, bspline_field)
        plot_sensors_map(bspline_field.axis_size, links=link_sampler.sensors, color="black")
        plt.tight_layout()
        plt.xlim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.ylim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.savefig(f"analysis/results/sensor_grid_link_only.svg")
        plt.show()
    # Random mixer
    mse_random_mixer_list = []
    for j in tqdm(range(n_mc)):
        gauges = slg_gauge.generate_gauge_position(is_center_uniform=False)
        mixer_sampler = signal_model.generate_mixed_sampler(bspline_field, gauges, links, sampler_cfg_gauge,
                                                            sampler_cfg_link)
        _mse_mixer, bcrb = compute_mse(bound_type, mixer_sampler, parameter, projection2mse)
        mse_random_mixer_list.append(_mse_mixer)
        if j == 0 and debug_plot:
            field = bcrb.diagonal().reshape([n_projection, n_projection])
            plot_bcrb_diagional_map(n_projection, field, bspline_field)
            plot_sensors_map(bspline_field.axis_size, gauges=gauges, links=link_sampler.sensors, color="black")
            plt.tight_layout()
            plt.xlim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
            plt.ylim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
            plt.savefig(f"analysis/results/sensor_random.svg")
            plt.show()
    mse_random_mixer_mean = np.mean(mse_random_mixer_list)
    mse_random_mixer_std = np.std(mse_random_mixer_list)

    # Optimal mixer
    gauges_optimal = find_point_sensors(n_gauge_sensors, n_projection, link_sampler, bspline_field, projection2mse,
                                        sampler_cfg_gauge, parameter.get_theta(), debug_plot=debug_plot)
    mixer_sampler = signal_model.generate_mixed_sampler(bspline_field, gauges_optimal, links, sampler_cfg_gauge,
                                                        sampler_cfg_link)
    mse_optimal_mixer, _ = compute_mse(bound_type, mixer_sampler, parameter, projection2mse)
    return mse_optimal_mixer, mse_random_mixer_mean, mse_random_mixer_std, mse_grid


def compute_mse(bound_type, mixer_sampler, parameter, projection2mse):
    if bound_type == BoundTypes.CRB:
        mixer_fim, _, _ = bounds.compute_fisher_information_matrix(mixer_sampler, parameter.get_theta())
    else:
        mixer_fim, _, _ = bounds.compute_bayesian_fisher_information_matrix(mixer_sampler, parameter.get_prior())
    mse_mixer, crb_gauge = projection2mse(mixer_fim)
    return mse_mixer, crb_gauge
