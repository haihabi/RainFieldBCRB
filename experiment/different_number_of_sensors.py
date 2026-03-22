import numpy as np
from tqdm import tqdm

import bounds
import signal_model
from constants import MISSPECIFIED, GAUGE, CML, X_AXIS
from experiment.rain_bounds import compute_all_bounds


def run_different_numer_of_sensor(slg, bspline_generator, n_projection, bound_type, sampler_cfg_gauge,
                                  sampler_cfg_link,
                                  link_length, parameter, is_random_placement=False, order=3, n_knots_base=8, n_mc=1,
                                  n_sensors_scales=[2, 2.5, 3, 4, 5],
                                  n_sensors_base=1):
    bspline_field = bspline_generator.generate_filed(order, n_knots_base=n_knots_base)
    projection2mse = bounds.Projection2MSE(bspline_field, n_projection, bounds.ProjectionType.FieldRMSE)
    fail_count = 0
    n_sensors_scales = np.asarray(n_sensors_scales)
    results_dict = {MISSPECIFIED: [],
                    GAUGE: [],
                    CML: [],
                    X_AXIS: np.round(n_sensors_scales * n_sensors_base)}
    for _ in range(len(n_sensors_scales)):
        results_dict[GAUGE].append([])
        results_dict[CML].append([])
        results_dict[MISSPECIFIED].append([])
    for _ in tqdm(range(n_mc)):
        links, gauges, mean_length,lengths = slg.generate_cmls_and_gauges(link_mean_length=link_length,
                                                                  n_sensors=int(np.max(n_sensors_scales) * n_sensors_base))
        for i,n_sensors_scale in enumerate(n_sensors_scales):
            n_sensors=int(n_sensors_scale * n_sensors_base)
            _links=links[:n_sensors, :]
            _gauges=gauges[:n_sensors, :]
            _mean_length=np.mean(lengths[:n_sensors])
            try:
                gauge_sampler = signal_model.generate_sampler(bspline_field, _gauges, sampler_cfg_gauge)
                link_sampler = signal_model.generate_sampler(bspline_field, _links, sampler_cfg_link)
                miss_sampler = signal_model.generate_sampler(bspline_field, _gauges, sampler_cfg_link)

                mse_gauge, mse_link, mse_mcrb = compute_all_bounds(bound_type, bspline_field,
                                                                   parameter, projection2mse,
                                                                   gauge_sampler, link_sampler,miss_sampler,_mean_length)
                results_dict[GAUGE][i].append(mse_gauge)
                results_dict[CML][i].append(mse_link)
                results_dict[MISSPECIFIED][i].append(mse_mcrb)

            except Exception as e:
                results_dict[GAUGE][i].append(np.nan)
                results_dict[CML][i].append(np.nan)
                results_dict[MISSPECIFIED][i].append(np.nan)
                fail_count += 1
    print("Failed to compute results for {} times".format(fail_count))
    return results_dict
