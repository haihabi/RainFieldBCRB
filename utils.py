import numpy as np

from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

rng = np.random.default_rng(2021)
MAP_SHAPE = [20, 20]  # [[-10,10], [-10,10]]
X, Y = np.mgrid[(-MAP_SHAPE[0] / 2):(MAP_SHAPE[0] / 2):100j, (-MAP_SHAPE[1] / 2): (MAP_SHAPE[1] / 2):100j]
LINK_MEAN_LENGTH = 2  # m
N_SENSORS_PER_AXIS = 3
GAUGE_NOISE = 0.001 * (MAP_SHAPE[0] / N_SENSORS_PER_AXIS)
SIGMA_SENSOR = 0.1
LINK_PARAMS = ['length', 'angle', 'center_x', 'center_y']
GAUGE_PARAMS = ['center_x', 'center_y']
BSPLINES_DEGREE = 3



def plot_sensors_map(axis_size, gauges=None, links=None, color="black"):
    """
    plots all sensors
    :param gauges: numpy array, [n_sensors, 2], gauges array
    :param links: numpy array, [n_sensors, 4], links array
    :return:
    """
    # f, ax = plt.subplots()
    if gauges is not None:
        plt.scatter(gauges[:, 0], gauges[:, 1], color=color,s=40)
    if links is not None:
        for link in links:
            plt.plot(link[:2], link[-2:], color=color,linewidth=4)

    plt.xlim((-axis_size / 2 - 1), (axis_size / 2 + 1))
    plt.ylim((-axis_size / 2 - 1), (axis_size / 2 + 1))
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

