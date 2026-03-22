import pickle

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import nanmean_after_z_score
from constants import CML, GAUGE, MISSPECIFIED, X_AXIS

if __name__ == '__main__':

    plt.rcParams.update({'font.size': 14})
    pickle_file= '/data/projects/RainFieldMisspecifiedCRB/results/results_order_8_random.pkl'
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    threshold=np.inf
    res_cml = nanmean_after_z_score(np.asarray(results[CML]), axis=1, threshold=threshold)
    res_gague = nanmean_after_z_score(np.asarray(results[GAUGE]), axis=1, threshold=threshold)
    res_miss = nanmean_after_z_score(np.asarray(results[MISSPECIFIED]), axis=1, threshold=threshold)

    plt.plot(results[X_AXIS], res_gague, label="Gauge")
    plt.plot(results[X_AXIS], res_cml, label="CML")
    plt.plot(results[X_AXIS], res_miss, label="Misspecified CML")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Order of B-spline")
    plt.ylabel("RMSE [mm/h]")
    plt.tight_layout()
    plt.savefig("results_rain_rate.svg")
    plt.show()
