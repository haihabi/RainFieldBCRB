import signal_model
from bounds.parameter import Parameter
from constants import MISSPECIFIED, GAUGE, CML, X_AXIS
from tqdm import tqdm

from experiment.rain_bounds import compute_all_bounds
import numpy as np


def run_different_length(link_length_scale, slg, rp, sampler_cfg_gauge, sampler_cfg_link, projection2mse,
                         bspline_field):
    results_dict = {MISSPECIFIED: [],
                    GAUGE: [],
                    CML: [],
                    X_AXIS: link_length_scale * rp.link_avg_length}
    for _ in range(len(link_length_scale)):
        results_dict[GAUGE].append([])
        results_dict[CML].append([])
        results_dict[MISSPECIFIED].append([])
    fail_count = 0
    parameter = Parameter(rp.n_parameter, rp.avg_rain_rate, variance=rp.rain_rate_std ** 2)
    for _ in tqdm(range(rp.n_mc)):
        try:
            links_base, gauges, mean_length, lengths = slg.generate_cmls_and_gauges(
                link_mean_length=rp.link_avg_length,
                lengths=np.ones(rp.n_sensors) * rp.link_avg_length,
                n_sensors=rp.n_sensors)



            for i, length_scale in enumerate(link_length_scale):
                sampler_cfg_gauge = signal_model.SamplerConfig(0.001 / (rp.link_avg_length * length_scale), 0.0,
                                                               signal_model.LineNormalization.L)
                gauge_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_gauge)

                lengths_scaled = lengths * length_scale
                links = signal_model.scale_link_length(links_base, lengths_scaled)
                link_sampler = signal_model.generate_sampler(bspline_field, links, sampler_cfg_link)
                miss_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_link)


                _mse_gauge, _mse_link, _mse_mcrb = compute_all_bounds(rp.bound_type, bspline_field,
                                                                      parameter, projection2mse,
                                                                      gauge_sampler, link_sampler, miss_sampler,
                                                                      None)

                results_dict[GAUGE][i].append(_mse_gauge)
                results_dict[CML][i].append(_mse_link)
                results_dict[MISSPECIFIED][i].append(_mse_mcrb)
        except Exception as e:
            fail_count += 1
    print("Failed to compute results for {} times".format(fail_count))
    return results_dict
