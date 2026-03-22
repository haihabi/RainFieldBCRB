import matplotlib.pyplot as plt

import signal_model
import numpy as np
from utils import plot_sensors_map

axis_size = 16
n_sensors = 15
n_parameter = 8 ** 2
fg = signal_model.BSplineRainFieldGenerator(axis_size, n_parameter)
slg = signal_model.SensorGenerator(map_shape=(axis_size / 2, axis_size / 2), n_sensors=n_sensors)
gauges = slg.generate_gauge_position(is_center_uniform=False)
links = slg.generate_cmls(3.4, gauges)
theta_base = np.random.rand(n_parameter)
theta_base /= np.linalg.norm(theta_base)
theta_base *= 10 / np.max(theta_base)

for i, order in enumerate([0,3]):
    field = fg.generate_filed(order)
    field.plot_field(theta_base)
    plt.xlabel("x[km]")
    plt.ylabel("y[km]")
    plt.xlim([-axis_size / 2, axis_size / 2])
    plt.ylim([-axis_size / 2, axis_size / 2])
    plt.tight_layout()

    plt.savefig(f"b-spline_field_{order}.svg")
    plt.show()


