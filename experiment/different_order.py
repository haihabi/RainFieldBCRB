import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import bounds
import signal_model
from array_design.find_point_sensor_placement import plot_bcrb_diagional_map
from config import BoundTypes
from constants import ORDER_LIST, MISSPECIFIED, GAUGE, CML, X_AXIS
from experiment.rain_bounds import compute_all_bounds
from utils import plot_sensors_map


def run_different_orders_fixed_sensors(slg, bspline_generator, n_projection, bound_type, sampler_cfg_gauge,
                                       sampler_cfg_link,
                                       link_length, parameter, is_random_placement=False, order_list=None):
    if order_list is None:
        order_list = ORDER_LIST
    results = []
    gauges = slg.generate_gauge_position(is_center_uniform=not is_random_placement)
    angles = np.pi * np.random.uniform(0, 1, gauges.shape[0])
    links, mean_length_act = slg.generate_cmls(gauges=gauges, link_mean_length=link_length, angles=angles,
                                               lengths=None)
    # n_mc = n_mc if is_random_placement else 1
    for order in tqdm(order_list):
        bspline_field = bspline_generator.generate_filed(order)
        print(bspline_field.x_set, bspline_field.n_parameters())
        projection2mse = bounds.Projection2MSE(bspline_field, n_projection, bounds.ProjectionType.Trace)
        # _results = []
        # for i in tqdm(range(n_mc)):

        gauge_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_gauge)
        link_sampler = signal_model.generate_sampler(bspline_field, links, sampler_cfg_link)
        if bound_type == BoundTypes.CRB:
            gauge_fim = bounds.compute_fisher_information_matrix(gauge_sampler, parameter.get_theta())
            link_fim = bounds.compute_fisher_information_matrix(link_sampler, parameter.get_theta())
        else:
            gauge_fim, _, _ = bounds.compute_bayesian_fisher_information_matrix(gauge_sampler, parameter.get_prior(),
                                                                                scale=np.sqrt(
                                                                                    mean_length_act))  # The scale is used to adjust the noise level to match the link noise level.
            link_fim, _, _ = bounds.compute_bayesian_fisher_information_matrix(link_sampler,
                                                                               parameter.get_prior())
        mse_gauge, crb_gauge = projection2mse(gauge_fim)
        mse_link, crb_link = projection2mse(link_fim)
        n_projection_act = projection2mse.actual_points()
        plt.subplot(1, 2, 1)
        field = crb_gauge.diagonal().reshape([n_projection_act, n_projection_act])
        plot_bcrb_diagional_map(n_projection_act, field, bspline_field)
        plot_sensors_map(bspline_field.axis_size, gauges=gauges, links=None, color="black")
        plt.tight_layout()
        plt.xlim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.ylim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.title("Point Sensors")
        # plt.show()
        plt.subplot(1, 2, 2)
        field = crb_link.diagonal().reshape([n_projection_act, n_projection_act])
        plot_bcrb_diagional_map(n_projection_act, field, bspline_field)
        plot_sensors_map(bspline_field.axis_size, gauges=None, links=links, color="black")
        plt.tight_layout()
        plt.xlim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.ylim([-bspline_generator.axis_size / 2, bspline_generator.axis_size / 2])
        plt.title("Line Sensors")
        plt.show()
        # plot_sensors_map(2*slg.map_shape[0], gauges=gauges, links=links, color="black")
        # plt.show()
        # _results.append([mse_gauge,
        #                  mse_link])
        results.append([[mse_gauge,
                         mse_link]])
    return np.asarray(results)


def run_different_orders_random_sensors(slg, bspline_generator, n_projection, bound_type, sampler_cfg_gauge,
                                        sampler_cfg_link,
                                        link_length, parameter, order_list, n_mc=100):
    results_dict = {MISSPECIFIED: [],
                    GAUGE: [],
                    CML: [],
                    X_AXIS: order_list}

    for order in tqdm(order_list):
        bspline_field = bspline_generator.generate_filed(order)
        projection2mse = bounds.Projection2MSE(bspline_field, n_projection, bounds.ProjectionType.FieldRMSE)
        _mse_gauge = []
        _mse_link = []
        _mse_mcrb = []
        for i in tqdm(range(n_mc)):
            links, gauges, mean_length = slg.generate_cmls_and_gauges(link_mean_length=link_length, n_sensors=None,
                                                                      is_center_uniform=False)

            gauge_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_gauge)
            link_sampler = signal_model.generate_sampler(bspline_field, links, sampler_cfg_link)

            mse_gauge, mse_link, mse_mcrb = compute_all_bounds(bound_type, bspline_field,
                                                               parameter, projection2mse,
                                                               gauge_sampler, link_sampler, gauge_sampler,mean_length)
            _mse_gauge.append(mse_gauge)
            _mse_link.append(mse_link)
            _mse_mcrb.append(mse_mcrb)

        results_dict[GAUGE].append(_mse_gauge)
        results_dict[CML].append(_mse_link)
        results_dict[MISSPECIFIED].append(_mse_mcrb)
    return results_dict
