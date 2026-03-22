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

    # pickle_file = '/data/projects/RainFieldMisspecifiedCRB/analysis/results/results_order_64_8_random.pkl'
    # pickle_file = '/data/projects/RainFieldMisspecifiedCRB/analysis/results/results_order_49_8_random.pkl'
    # n_s_array = [8, 16, 32, 49, 64, 96, 128]
    # cp_list=[]
    # for n_s in n_s_array:
    #     pickle_file = f'/data/projects/RainFieldMisspecifiedCRB/analysis/results/results_order_{n_s}_8_random.pkl'
    #     with open(pickle_file, 'rb') as f:
    #         field = pickle.load(f)
    #     size=5
    #     fig = plt.figure(figsize=(size*8.5/4, size), dpi=600)
    #     mp, ml = ploting_function(field)
    #     delta = mp - ml # point - link
    #     pos=np.max(np.where(delta>0)) # x1
    #     neg=pos+1 # x2
    #     pos_mp=mp[pos] # y1
    #     neg_mp=mp[neg] # y2
    #
    #     a=(neg_mp-pos_mp)/(neg-pos) # slope
    #     b=pos_mp-a*pos
    #
    #
    #     neg_ml=ml[neg]
    #     pos_ml = ml[pos]
    #     a_ml = (neg_ml - pos_ml) / (neg - pos)
    #     b_ml = pos_ml - a_ml * pos
    #
    #     x=(b_ml-b)/(a-a_ml)
    #     y=a*x+b
    #     print(x)
    #     cp_list.append(x)
    #
    #
    #     plt.plot([x], [y], marker='o', color='red',label='Crossing Point')
    #
    #
    #
    #     print(delta)
    #     print("a")
    #     plt.legend()
    #     # plt.title(f"n_sensors={n_s}")
    #     plt.tight_layout()
    #     plt.savefig(f"order_vs_rmse_{n_s}.svg")
    #     plt.show()
    # area=8*8
    # density=np.sqrt(area/np.array(n_s_array))
    #
    # plt.plot(density,cp_list)
    # plt.xlabel(r"Sensors Density $\delta$ [km]")
    # plt.ylabel("Order")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("order_vs_density.svg")
    # plt.show()
    # # print(np.mean(field[:, :, 0], axis=-1))
    # # print(np.mean(field[:, :, 1], axis=-1))
