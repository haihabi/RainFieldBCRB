import signal_model
from bounds.parameter import Parameter
from constants import MISSPECIFIED, GAUGE, CML, X_AXIS
from experiment.rain_bounds import compute_all_bounds
from tqdm import tqdm


def run_rain_rate(avg_rain_rate_var, slg, rp, bspline_field, sampler_cfg_gauge, sampler_cfg_link, projection2mse, n_mc,mu=0.1):
    results_dict = {MISSPECIFIED: [],
                    GAUGE: [],
                    CML: [],
                    X_AXIS: avg_rain_rate_var}
    for _ in range(len(avg_rain_rate_var)):
        results_dict[GAUGE].append([])
        results_dict[CML].append([])
        results_dict[MISSPECIFIED].append([])
    fail_count = 0
    for _ in tqdm(range(n_mc)):
        links, gauges, mean_length,_ = slg.generate_cmls_and_gauges(rp.link_avg_length, rp.n_sensors)
        gauge_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_gauge)
        link_sampler = signal_model.generate_sampler(bspline_field, links, sampler_cfg_link)
        try:
            for i, var in enumerate(avg_rain_rate_var):
                parameter = Parameter(rp.n_parameter, mu, variance=var**2)
                _mse_gauge, _mse_link, _mse_mcrb = compute_all_bounds(rp.bound_type, bspline_field,
                                                                      parameter, projection2mse,
                                                                      gauge_sampler, link_sampler,gauge_sampler,mean_length, True)
                results_dict[GAUGE][i].append(_mse_gauge)
                results_dict[CML][i].append(_mse_link)
                results_dict[MISSPECIFIED][i].append(_mse_mcrb)
        except Exception as e:
            fail_count += 1
    print("Failed to compute results for {} times".format(fail_count))
    return results_dict
