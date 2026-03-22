import numpy as np
from matplotlib import pyplot as plt
from sympy import integrate

from signal_model.spline_field.bspline_helpers import create_bsplines_set, create_bsplines_patch_set, lambdify_bspline
from signal_model.sensors_locations import SensorGenerator
from sympy.abc import x, y, t
from enum import Enum


class BSplineType(Enum):
    ORDEREDGEFILLED = 0
    ORDERKENOTS = 1


def plot_spline_field(in_theta, in_bsplines_set, in_axis_size, in_n_points):
    axis_size = in_axis_size * 1.4
    x_array = np.linspace(-axis_size / 2, axis_size / 2, in_n_points)
    xx, yy = np.meshgrid(x_array, x_array)
    field = np.zeros([in_n_points, in_n_points])
    for i, bspline in enumerate(in_bsplines_set):
        f = lambdify_bspline(bspline)
        field += f(xx, yy) * in_theta[i]
    z_min, z_max = np.min(field), np.abs(field).max()
    plt.pcolormesh(xx, yy, field, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.xlim([-axis_size / 2, axis_size / 2])
    plt.ylim([-axis_size / 2, axis_size / 2])
    plt.colorbar()


class BSplineRainField(object):
    def __init__(self, in_axis_size, in_order, knots):
        self.x_set = create_bsplines_set(x, in_order, knots)
        self.y_set = create_bsplines_set(y, in_order, knots)
        self.knots = knots
        self.patch_set = create_bsplines_patch_set(self.x_set, self.y_set)
        self.patch_function = {idx: lambdify_bspline(bspline) for idx, bspline in enumerate(self.patch_set)}
        self.axis_size = in_axis_size

    def n_parameters(self):
        return len(self.patch_set)

    def projection_matrix(self, n_points):
        points = SensorGenerator((self.axis_size / 2, self.axis_size / 2), n_sensors=n_points ** 2,
                                 is_center_uniform=True).generate_gauge_position(noise_level=0.0)
        return create_projection_matrix(self, points).T

    def plot_line(self, in_n_points=1000):
        x = np.linspace(np.min(self.knots), np.max(self.knots), in_n_points)
        field = np.zeros([in_n_points, len(self.x_set)])
        for i, bspline in enumerate(self.x_set):
            f = lambdify_bspline(bspline)
            field[:, i] = f(x)
        plt.plot(x, field)

    def plot_field(self, in_theta, in_n_points=None):
        if in_n_points is None:
            in_n_points = 2 * in_theta.shape[0]
        plot_spline_field(in_theta, self.patch_set, self.axis_size, in_n_points)


class BSplineRainFieldGenerator():
    def __init__(self, in_axis_size, n_parameters, fild_type=BSplineType.ORDEREDGEFILLED):
        self.n_parameters = n_parameters
        self.n_knots = int(np.sqrt(self.n_parameters))
        self.fild_type = fild_type
        self.axis_size = in_axis_size

    def generate_filed(self, in_order, n_knots_base=10) -> BSplineRainField:
        if self.fild_type == BSplineType.ORDEREDGEFILLED:
            knots = np.linspace(-self.axis_size / 2, self.axis_size / 2, n_knots_base)
            if len(np.unique(knots[:in_order + 1])) != 1:  # last and first knots multiplicty is always d
                knots = np.hstack((knots[0] * np.ones(in_order), knots))
            if len(np.unique(knots[-(in_order + 1):])) != 1:
                knots = np.hstack((knots, knots[-1] * np.ones(in_order)))
            knots = list(knots)
        elif self.fild_type == BSplineType.ORDERKENOTS:
            knots = np.linspace(-self.axis_size / 2, self.axis_size / 2, self.n_knots + 1)
            delta = np.diff(knots)[0]
            for _ in range(in_order):
                shift_knots = knots - delta / 2
                knots = np.concatenate([shift_knots, np.asarray([knots[-1] + delta / 2])])

            # n_knots = self.n_knots + in_order + 1
            # rep = 2 * in_order
            # knots = np.linspace(-self.axis_size / 2, self.axis_size / 2, n_knots - rep)
            # delta_knot = knots[1] - knots[0]
            # knots = np.concatenate([knots[0] * np.ones(in_order) - np.cumsum(np.ones(in_order) * delta_knot), knots,
            #                         np.ones(in_order) * knots[-1] + np.cumsum(np.ones(in_order) * delta_knot)])
        else:
            raise ValueError(f"Unknown BSplineType: {self.fild_type}")

        return BSplineRainField(self.axis_size, in_order, knots)


def create_projection_matrix(in_field: BSplineRainField, in_sensors, approximate=True,
                             n_approximation_point=10000):
    """
    Creates projection matrix of bspline patches over a set of sensors, either gauges, links or surfaces
    :param in_field:
    :param n_approximation_point:
    :param in_sensors: numpy array, having either [2, 4, 9] corresponding to either [gauges, links, surfaces]
    :param approximate: bool, whether to numerically calculate the projection matrix (speed)
    :return: proj_matrix: numpy array [n_sensors, n_bsplines] projection matrix
    """
    proj_matrix = np.zeros((in_sensors.shape[0], len(in_field.patch_set)))
    for idx, bspline in enumerate(in_field.patch_set):
        f = in_field.patch_function[idx]
        if in_sensors.shape[-1] == 2:  # gauges
            x_gauge = in_sensors[:, 0]
            y_gauge = in_sensors[:, 1]
            projection = f(x_gauge, y_gauge)
            proj_matrix[:, idx] = projection
        elif in_sensors.shape[-1] == 4:  # links
            for sensor_idx, sensor in enumerate(in_sensors):
                x_start = sensor[0]
                x_end = sensor[1]
                y_start = sensor[2]
                y_end = sensor[3]

                if not approximate:
                    a1tag = x_end - x_start
                    b1tag = x_start
                    a2tag = y_end - y_start
                    b2tag = y_start
                    bspline_new = bspline.subs([(x, a1tag * t + b1tag), (y, a2tag * t + b2tag)])
                    projection = integrate(bspline_new, (t, 0, 1))
                else:
                    x_array = np.linspace(x_start, x_end, n_approximation_point)
                    y_array = (x_array - x_start) * (y_end - y_start) / (x_end - x_start) + y_start
                    projection = np.mean(
                        f(x_array, y_array))

                proj_matrix[sensor_idx, idx] = projection
        else:
            raise ValueError('create_projection_matrix:: invalid sensors input')

    return proj_matrix
