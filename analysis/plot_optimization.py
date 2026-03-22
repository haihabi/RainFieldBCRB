import numpy as np
from matplotlib import pyplot as plt

from analysis.plot_field import n_sensors

a = [(np.float64(0.09053187543872399), np.float64(0.09677311873434487), np.float64(0.0033389710647599644),
      np.float64(0.0953485745366047)),
     (np.float64(0.06364953441612466), np.float64(0.07906652298278909), np.float64(0.0061165226210306645),
      np.float64(0.0795565556333606)),
     (np.float64(0.035852490637243106), np.float64(0.057618940290646105), np.float64(0.0035821966150995158),
      np.float64(0.0446033364133714)),
     (np.float64(0.017748163698277454), np.float64(0.040026537010896104), np.float64(0.005296841275518987),
      np.float64(0.02198392021584034)),
     (np.float64(0.009033458297147107), np.float64(0.029039363832328146), np.float64(0.00347063483615259),
      np.float64(0.01453557539506238)),
     (np.float64(0.006143310257525757), np.float64(0.018296402078284705), np.float64(0.00205255045214806),
      np.float64(0.008017859753716956))

     ]
a = np.asarray(a)
n_sensors = [1, 4, 9, 16, 25,36]
size=5
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(size * 8.5 / 4, size), dpi=600)

plt.plot(n_sensors, a[:, 0], label="Optimized")
plt.plot(n_sensors, a[:, 1], label="Random")
plt.plot(n_sensors, a[:, -1], label="Grid")
plt.yscale("log")
plt.legend()
plt.xlabel(r"Number of point sensors $N_{P}$ ")
plt.ylabel("RMSE[mm/h]")
plt.grid()
plt.tight_layout()
plt.savefig("sensor_optimization_n.svg")
plt.show()
