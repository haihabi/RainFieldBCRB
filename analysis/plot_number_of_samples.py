import pickle

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import nanmean_after_z_score
from constants import CML, GAUGE, X_AXIS,MISSPECIFIED

for o,n in zip([1,2,3],[7,6,5]):

    path=f"/data/projects/RainFieldMisspecifiedCRB/results/results_n_sensors_8_random_{o}_{n}_CRB_new.pkl" # "/data/projects/RainFieldMisspecifiedCRB/results/results_n_sensors_8_random_2_6.pkl"
    # path="/data/projects/RainFieldMisspecifiedCRB/results/results_n_sensors_8_random_2_6.pkl" # "/data/projects/RainFieldMisspecifiedCRB/results/results_n_sensors_8_random_2_6.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    # print("a")
    # results[:,:,0]
    # print(np.asarray(results[CML]).shape)
    res_cml,std_cml = nanmean_after_z_score(np.asarray(results[CML]), axis=1, threshold=3)
    res_gague,std_gauge = nanmean_after_z_score(np.asarray(results[GAUGE]), axis=1, threshold=3)
    res_miss,std_miss = nanmean_after_z_score(np.asarray(results[MISSPECIFIED]), axis=1, threshold=3)

    # print(results[X_AXIS]/2)
    plt.plot(results[X_AXIS], res_gague, label="CRB:Point Sensors")
    plt.plot(results[X_AXIS], res_cml, label="CRB:Line Sensors")
    plt.plot(results[X_AXIS], res_miss, label="MCRB:Line Sensors")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Number of sensors")
    plt.ylabel("RMSE [mm/hr]")
    plt.tight_layout()
    plt.savefig("results_n_sensors.svg")
    plt.show()


    plt.plot(results[X_AXIS], res_miss-res_cml, label="CRB:Point Sensors")
    plt.show()
    delta=res_miss-res_gague
    print(delta[0])