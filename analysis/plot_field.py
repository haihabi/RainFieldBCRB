import matplotlib.pyplot as plt

import signal_model
import numpy as np

from signal_model.spline_field import BSplineType
from utils import plot_sensors_map

axis_size = 16
n_sensors = 15
n_parameter = 8 ** 2
fg = signal_model.BSplineRainFieldGenerator(axis_size, n_parameter, fild_type=BSplineType.ORDERKENOTS)
slg = signal_model.SensorGenerator(map_shape=(axis_size / 2, axis_size / 2), n_sensors=n_sensors,is_center_uniform=True)
gauges = slg.generate_gauge_position()
links = slg.generate_cmls(3.4, gauges)
theta_base = np.random.rand(n_parameter)
theta_base /= np.linalg.norm(theta_base)
theta_base *= 10 / np.max(theta_base)

for i, order in enumerate([0,3]):
    field = fg.generate_filed(order)
    field.plot_field(theta_base)
    plt.xlabel("x[km]",fontsize=16)
    plt.ylabel("y[km]",fontsize=16)
    plt.xlim([-axis_size / 2, axis_size / 2])
    plt.ylim([-axis_size / 2, axis_size / 2])
    plt.tight_layout()

    plt.savefig(f"./images/b-spline_field_{order}.svg")
    plt.show()


