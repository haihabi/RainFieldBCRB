import pickle

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import nanmean_after_z_score
from constants import CML, GAUGE, X_AXIS,MISSPECIFIED




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

