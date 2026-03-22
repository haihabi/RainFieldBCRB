import bounds
import signal_model
from tqdm import tqdm

from constants import MISSPECIFIED, MIXER, X_AXIS


def misspecified_sensors_mixer(slg, bspline_field, parameter, projection2mse, link_length, n_sensors, sampler_cfg_gauge,
                               sampler_cfg_link, n_mc, gauge_rate_array, resample=False):
    results_dict = {MISSPECIFIED: [],
                    MIXER: [],
                    X_AXIS: gauge_rate_array}
    for _ in range(len(gauge_rate_array)):
        results_dict[MIXER].append([])
        results_dict[MISSPECIFIED].append([])

    theta = parameter.get_theta(bspline_field.n_parameters(), resample=resample)
    fail_count = 0
    for _ in tqdm(range(n_mc)):
        links, gauges, _,_ = slg.generate_cmls_and_gauges(link_mean_length=link_length,
                                                                  n_sensors=n_sensors)
        # misspecifed_sampler = signal_model.generate_sampler(bspline_field, gauges, sampler_cfg_link)
        try:
            for i, p in enumerate(gauge_rate_array):
                n_gauge = int(len(gauges) * p)
                misspecifed_sampler = signal_model.generate_mixed_sampler(bspline_field, gauges[:n_gauge,:], gauges[n_gauge:,:], sampler_cfg_gauge,
                                                                    sampler_cfg_link)

                mixer_sampler = signal_model.generate_mixed_sampler(bspline_field, gauges[:n_gauge,:], links[n_gauge:,:], sampler_cfg_gauge,
                                                                    sampler_cfg_link)

                link_crb = bounds.compute_fisher_information_matrix(mixer_sampler, theta)
                link_mcrb = bounds.compute_mcrb(mixer_sampler, misspecifed_sampler, theta, parameter.get_covariance(),
                                                parameter.get_mean())

                mse_miss, _ = projection2mse(link_mcrb)
                mse_mixer, _ = projection2mse(link_crb)

                results_dict[MIXER][i].append(mse_mixer)

                results_dict[MISSPECIFIED][i].append(mse_miss)
        except Exception as e:
            fail_count += 1
    print("Failed to compute results for {} times".format(fail_count))
    return results_dict
