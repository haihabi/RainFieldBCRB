import dataclasses
import random

import numpy as np

import signal_model

from bounds.parameter import Parameter
from config import BoundTypes
from experiment.different_length import run_different_length
from experiment.different_number_of_sensors import run_different_numer_of_sensor
from experiment.different_order import run_different_orders_random_sensors
from experiment.different_rain_rate import run_rain_rate
from experiment.misspecified_sensors_mixer import misspecified_sensors_mixer
from experiment.sensor_mixer import run_sensor_mixer
import pickle
from signal_model import spline_field
from tqdm import tqdm
from enum import Enum
import bounds


class RunType(Enum):
    Order = 0
    Length = 1
    RainRate = 2
    Location = 3
    NSensors = 5
    OptimizedPointVsNSensors = 4
    Mixer = 6


def mse2db(in_mse):
    return 10 * np.log10(in_mse)


@dataclasses.dataclass
class RunParameters:
    axis_size: float
    min_length: float
    max_length: float
    link_avg_length: float
    n_parameter: int
    n_sensors: int
    avg_rain_rate: float
    rain_rate_std:float=np.sqrt(2.8)
    bound_type: BoundTypes = BoundTypes.CRB
    is_random_placement: bool = True
    order: int = 1
    n_knots_base: int = 6
    parameter_variance: float = 2.8
    n_mc:int = 100


def set_seed():
    np.random.seed(1)
    random.seed(0)


def dump_results(_results_dict,_run_type,_rp,_sensors_placement_type):
    print("Saving results to file")
    with open(f"results/results_{_run_type.name}_{_rp.axis_size}_{_sensors_placement_type}_{_rp.order}_{_rp.n_knots_base}_{_rp.n_sensors}_{_rp.bound_type.name}.pkl","wb") as f:
        pickle.dump(_results_dict, f)


if __name__ == '__main__':
    print("Run main experiment")
    set_seed()

    run_type = RunType.Length
    rp = RunParameters(axis_size=8,
                       min_length=0.3,
                       max_length=50,
                       link_avg_length=3.5,
                       order=2,
                       n_knots_base=6,
                       n_parameter=7 ** 2,
                       n_sensors= 2*7 ** 2,  # 4 * 7 ** 2
                       avg_rain_rate=0.1,
                       bound_type=BoundTypes.CRB,
                       is_random_placement=True)
    sensors_placement_type = "random" if rp.is_random_placement else "deterministic"

    parameter = Parameter(rp.n_parameter, rp.avg_rain_rate, variance=2.8)
    sampler_cfg_gauge = signal_model.SamplerConfig(0.001/rp.link_avg_length, 0.0, signal_model.LineNormalization.L)
    sampler_cfg_link = signal_model.SamplerConfig(0.001, 0.00, signal_model.LineNormalization.SQRTL)

    bspline_generator = spline_field.BSplineRainFieldGenerator(rp.axis_size, rp.n_parameter, fild_type=
    spline_field.BSplineType.ORDEREDGEFILLED)
    slg = signal_model.SensorGenerator(map_shape=(rp.axis_size / 2, rp.axis_size / 2), n_sensors=rp.n_sensors,
                                       is_center_uniform=not rp.is_random_placement)

    n_projection = int(1 * np.ceil(np.sqrt(rp.n_parameter)))
    bspline_field = bspline_generator.generate_filed(rp.order, rp.n_knots_base)
    projection2mse = bounds.Projection2MSE(bspline_field, n_projection, bounds.ProjectionType.FieldRMSE)
    if run_type == RunType.NSensors:
        # 0.125 / 8, 0.125 / 4,0.125 / 2, 0.125, 0.25, 0.5, 1,
        n_sensors_scales = [  2, 2.5,2.75, 3, 4, 5]
        results = run_different_numer_of_sensor(slg, bspline_generator, n_projection, rp.bound_type,
                                                sampler_cfg_gauge,
                                                sampler_cfg_link,
                                                rp.link_avg_length, parameter,
                                                is_random_placement=rp.is_random_placement, order=rp.order,
                                                n_knots_base=rp.n_knots_base,
                                                n_sensors_scales=n_sensors_scales,
                                                n_sensors_base=rp.n_sensors,
                                                n_mc=rp.n_mc)
        dump_results(results,run_type,rp,sensors_placement_type)
    elif run_type == RunType.Order:
        order_list = [0, 1, 2, 3, 4, 5]
        results = run_different_orders_random_sensors(slg, bspline_generator, n_projection, rp.bound_type,
                                                      sampler_cfg_gauge,
                                                      sampler_cfg_link,
                                                      rp.link_avg_length, parameter, order_list,
                                                      n_mc=100)

        with open(f"results/results_order_{rp.axis_size}_{run_type}.pkl", "wb") as f:
            pickle.dump(results, f)
    elif run_type == RunType.OptimizedPointVsNSensors:
        slg_link = signal_model.SensorGenerator(
            map_shape=(bspline_generator.axis_size / 2, bspline_generator.axis_size / 2), n_sensors=6)
        link_center = slg_link.generate_gauge_position(is_center_uniform=False)
        angles = np.pi * np.random.uniform(0, 1, link_center.shape[0])
        links, _ = slg_link.generate_cmls(gauges=link_center, link_mean_length=rp.link_avg_length, angles=angles)
        for n_gauge in [4]:
            res = run_sensor_mixer(links, n_gauge, bspline_generator, n_projection, rp.link_avg_length,
                                   sampler_cfg_gauge,
                                   sampler_cfg_link, parameter, debug_plot=True)

    elif run_type == RunType.Length:
        scale_array = np.linspace(0.1/3.5, 1.0, 10)
        results_dict=run_different_length(scale_array, slg, rp, sampler_cfg_gauge, sampler_cfg_link, projection2mse, bspline_field)
        dump_results(results_dict,run_type,rp,sensors_placement_type)
    elif run_type == RunType.RainRate:
        avg_rain_rate_std = np.linspace(0.01, 8, 20)
        results_dict = run_rain_rate(avg_rain_rate_std, slg, rp, bspline_field, sampler_cfg_gauge, sampler_cfg_link,
                                     projection2mse, rp.n_mc)
        dump_results(results_dict,run_type,rp,sensors_placement_type)

    elif run_type == RunType.Mixer:
        gauge_rate_array = np.linspace(0.0, 1, 2)
        results_dict = misspecified_sensors_mixer(slg, bspline_field, parameter, projection2mse, rp.link_avg_length,
                                                  rp.n_sensors,
                                                  sampler_cfg_gauge,
                                                  sampler_cfg_link, rp.n_mc, gauge_rate_array, resample=False)
        dump_results(results_dict,run_type,rp,sensors_placement_type)

