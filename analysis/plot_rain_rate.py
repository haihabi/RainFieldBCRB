import pickle

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import nanmean_after_z_score
# from analysis.plot_order import ploting_function
from constants import CML, GAUGE, X_AXIS,MISSPECIFIED


# def nanmean_after_z_score(results, axis=1, threshold=3):
#     mu = np.nanmean(results, axis=axis, keepdims=True)
#     std = np.nanstd(results, axis=axis, keepdims=True)
#     z_score = (np.abs(results - mu)) / std
#     mask = z_score < threshold
#     masked_results = np.where(mask, results, np.nan)
#     return np.nanmean(masked_results, axis=axis)


with open("/data/projects/RainFieldMisspecifiedCRB/results/results_rain_rate_8_random_2_6_196_new.pkl", "rb") as f:
    results = pickle.load(f)
threshold=3
res_cml,_ = nanmean_after_z_score(np.asarray(results[CML]), axis=1, threshold=threshold)
res_gague,_ = nanmean_after_z_score(np.asarray(results[GAUGE]), axis=1, threshold=threshold)
res_miss,_ = nanmean_after_z_score(np.asarray(results[MISSPECIFIED]), axis=1, threshold=threshold)

plt.plot(results[X_AXIS], res_gague, label="CRB:Point Sensors")
plt.plot(results[X_AXIS], res_cml, label="CRB:Line Sensors")
plt.plot(results[X_AXIS], res_miss, label="MCRB:Line Sensors")
plt.legend()
plt.yscale("log")
plt.grid()
plt.xlabel("$\sigma_{\psi}$ [mm/h]")
plt.ylabel("RMSE [mm/h]")
plt.tight_layout()
plt.savefig("results_rain_rate.svg")
plt.show()

