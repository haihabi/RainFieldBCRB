import pickle

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import nanmean_after_z_score
# from analysis.plot_order import ploting_function
from constants import CML, GAUGE, X_AXIS,MISSPECIFIED,MIXER




with open("/data/projects/RainFieldMisspecifiedCRB/results/results_mixer_8_random_2_6_196.pkl", "rb") as f:
    results = pickle.load(f)
threshold=np.inf
res_cml,_ = nanmean_after_z_score(np.asarray(results[MIXER]), axis=1, threshold=threshold)
res_miss,_ = nanmean_after_z_score(np.asarray(results[MISSPECIFIED]), axis=1, threshold=threshold)

plt.plot(results[X_AXIS], res_cml, label="CRB")
plt.plot(results[X_AXIS], res_miss, label="MCRB")
plt.legend()
plt.yscale("log")
plt.grid()
plt.xlabel("$\sigma_{\psi}$ [mm/h]")
plt.ylabel("RMSE [mm/h]")
plt.tight_layout()
plt.savefig("results_mixer.svg")
plt.show()

