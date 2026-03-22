import numpy as np

LINK_MEAN_LENGTH = 2
N_SENSORS_PER_AXIS = 3


def scale_link_length(links, length_scale):
    x_center = (links[:, 0] + links[:, 1]) / 2
    y_center = (links[:, 2] + links[:, 3]) / 2
    lengths = length_scale * np.sqrt((links[:, 0] - links[:, 1]) ** 2 + (links[:, 2] - links[:, 3]) ** 2)
    angles= np.arctan2(links[:, 3] - links[:, 2], links[:, 1] - links[:, 0])
    x_start = x_center - np.cos(angles) * lengths / 2
    x_end = x_center + np.cos(angles) * lengths / 2
    y_start = y_center - np.sin(angles) * lengths / 2
    y_end = y_center + np.sin(angles) * lengths / 2
    links = np.vstack((x_start, x_end, y_start, y_end)).T
    return links


def selected_point_on_links(links, n_points=1):
    if n_points == 1:
        return (links[:, :2] + links[:, 2:]) / 2
    else:
        points = []
        for i in range(links.shape[0]):
            x_start, x_end, y_start, y_end = links[i]
            x_points = np.linspace(x_start, x_end, n_points)
            y_points = np.linspace(y_start, y_end, n_points)
            points.append(np.vstack((x_points, y_points)).T)
        return np.vstack(points)


class SensorGenerator(object):
    def __init__(self, map_shape, is_center_uniform, n_sensors=N_SENSORS_PER_AXIS):
        self.map_shape = map_shape
        self.n_sensors = n_sensors
        if len(map_shape) != 2:
            raise ValueError("Map shape must be 2D")
        self.is_center_uniform = is_center_uniform

    def generate_gauge_position(self, n_sensors=None, noise_level=0.0):
        n_sensors = self.n_sensors if n_sensors is None else n_sensors
        if self.is_center_uniform:
            m = int(np.ceil(np.sqrt(n_sensors)))
            x_grid = np.linspace(-self.map_shape[0], self.map_shape[0], m)
            y_grid = np.linspace(-self.map_shape[1], self.map_shape[1], m)

            gauges = np.dstack(np.meshgrid(x_grid, y_grid)).reshape(-1, 2)
            gauges = gauges+np.random.randn(*gauges.shape)*noise_level
        else:
            x_gauge = np.random.uniform(-self.map_shape[0], self.map_shape[0], n_sensors)
            y_gauge = np.random.uniform(-self.map_shape[1], self.map_shape[1], n_sensors)
            gauges = np.array((x_gauge, y_gauge)).T
        return gauges

    def generate_cmls(self, link_mean_length=LINK_MEAN_LENGTH, gauges=None, angles=None,
                      lengths=None):
        if gauges is None:
            gauges = self.generate_gauge_position()
        # create links
        if angles is None:
            angles = np.pi * np.random.uniform(0, 1, gauges.shape[0])
        if lengths is None:
            lengths = np.random.exponential(link_mean_length, gauges.shape[0])
            lengths = np.clip(lengths, 0.3, 50)
        x_start = gauges[:, 0] - np.cos(angles) * lengths / 2
        x_end = gauges[:, 0] + np.cos(angles) * lengths / 2
        y_start = gauges[:, 1] - np.sin(angles) * lengths / 2
        y_end = gauges[:, 1] + np.sin(angles) * lengths / 2
        links = np.vstack((x_start, x_end, y_start, y_end)).T
        return links, np.mean(lengths),lengths

    def generate_cmls_and_gauges(self, link_mean_length=LINK_MEAN_LENGTH, n_sensors=None,
                                 angles=None, lengths=None):
        gauges = self.generate_gauge_position(n_sensors=n_sensors)
        links, mean_length,lengths = self.generate_cmls(link_mean_length=link_mean_length, gauges=gauges, angles=angles,
                                                lengths=lengths)
        return links, gauges, mean_length,lengths
