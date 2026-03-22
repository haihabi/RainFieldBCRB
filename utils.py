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
# KNOTS = np.linspace(-MAP_SHAPE[0] / 2, MAP_SHAPE[0] / 2, 10)


# def generate_sensors_map(map_shape=MAP_SHAPE, n_sensors=N_SENSORS_PER_AXIS, link_mean_length=LINK_MEAN_LENGTH,
#                          is_gauge_uniform=False):
#     """
#     Generates sensors on a grid , including n_sensors^2 sensors
#     Gauges are either uniformly generated on a grid with small translations or uniformly distributed inside the map
#     Links are generated so there is a gauge in their middle
#     :param map_shape: list, [map_shape_x, map_shape_y]
#     :param n_sensors: number of sensors to be generated per axis
#     :param link_mean_length: mean link length (for exponential distribution)
#     :return: gauges: numpy array, [n_sensors, 2] where each row has the following structure: (x, y)
#              links: numpy array, [n_sensors, 4] where each row has the following structure: (x_start, x_end, y_start, y_end)
#              surfaces: numpy array, [n_sensors, 9] where each row has the following structure: (xbl, xbr, xul, xur, ybl, ybr, yul, yur, angle)
#     """
#     # create gauges
#     if is_gauge_uniform:
#         x_grid = np.linspace(-map_shape[0] / 2 + 1, map_shape[0] / 2 - 1, n_sensors)
#         y_grid = np.linspace(-map_shape[1] / 2 + 1, map_shape[1] / 2 - 1, n_sensors)
#         gauges = np.dstack(np.meshgrid(x_grid, y_grid)).reshape(-1, 2)
#         gauges_noise = rng.normal(0, GAUGE_NOISE, (gauges.shape[0], gauges.shape[1]))
#         gauges = gauges + gauges_noise
#     else:
#         x_gauge = rng.uniform(-map_shape[0] / 2, map_shape[0] / 2, n_sensors)
#         y_gauge = rng.uniform(-map_shape[1] / 2, map_shape[1] / 2, n_sensors)
#         gauges = np.array((x_gauge, y_gauge)).T
#
#     # create links
#     angles = 2 * np.pi * rng.uniform(0, 1, gauges.shape[0])
#     lengths = rng.exponential(link_mean_length, gauges.shape[0])
#     x_start = gauges[:, 0] - np.cos(angles) * lengths / 2
#     x_end = gauges[:, 0] + np.cos(angles) * lengths / 2
#     y_start = gauges[:, 1] - np.sin(angles) * lengths / 2
#     y_end = gauges[:, 1] + np.sin(angles) * lengths / 2
#     links = np.vstack((x_start, x_end, y_start, y_end)).T
#
#     # create surfaces
#     second_length = rng.exponential(link_mean_length, gauges.shape[0])
#     xbl = gauges[:, 0] - np.cos(angles) * lengths / 2 - np.sin(angles) * second_length / 2
#     ybl = gauges[:, 1] - np.sin(angles) * lengths / 2 + np.cos(angles) * second_length / 2
#     xbr = gauges[:, 0] - np.cos(angles) * lengths / 2 + np.sin(angles) * second_length / 2
#     ybr = gauges[:, 1] - np.sin(angles) * lengths / 2 - np.cos(angles) * second_length / 2
#
#     xul = gauges[:, 0] + np.cos(angles) * lengths / 2 - np.sin(angles) * second_length / 2
#     yul = gauges[:, 1] + np.sin(angles) * lengths / 2 + np.cos(angles) * second_length / 2
#     xur = gauges[:, 0] + np.cos(angles) * lengths / 2 + np.sin(angles) * second_length / 2
#     yur = gauges[:, 1] + np.sin(angles) * lengths / 2 - np.cos(angles) * second_length / 2
#     surfaces = np.vstack((xbl, xbr, xul, xur, ybl, ybr, yul, yur, angles)).T
#
#     return gauges, links, surfaces


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


def calc_bound(H, reparametrize, sigma=SIGMA_SENSOR, flavour='coefficients'):
    """
    Takes a projection matrix and calculates the CRLB of the sensors scheme that created the matrix
    :param H: numpy array, [n_sensors, n_bsplines], projection matrix
    # :param patch_set: list, bsplines patch set
    :param reparametrize: method for reparameterization to accumulated rainfall. either ['crlb','fim']
    :param sigma: noise standrad deviation of sensors
    :param flavour: bound to which value. either ['accumulated', 'coefficients']
    :return: CRLB
    """
    fim_mat = np.matmul(H.T, H) * (1 / sigma ** 2)
    final_mat = fim_mat
    if reparametrize == 'crlb':
        crlb_mat = np.linalg.inv(fim_mat)
        final_mat = crlb_mat
    # if projection_matrix is not None:
    #     return np.trace(projection_matrix.T @ crlb_mat @ projection_matrix) / crlb_mat.shape[0]
    if flavour == 'coefficients':  # coefficients is directly retrieved from the crlb
        if reparametrize != 'crlb':
            raise ValueError('calc_bound:: cannot return bound on coefficients without crlb reparametrization')
        else:
            return np.trace(crlb_mat) / crlb_mat.shape[0]
    # elif flavour == 'accumulated':
    #     J = np.array([np.sum(lambdify_bspline(p)(X, Y)) for p in patch_set])
    #     if reparametrize == 'fim':
    #         J = 1 / J
    #         return 1 / (J.T @ final_mat @ J)
    #     elif reparametrize == 'crlb':
    #         return J.T @ final_mat @ J
    else:
        raise ValueError(f'calc_MSE:: {flavour=} is unsupported')

# from tqdm import tqdm
#
# if __name__ == '__main__':
#
#     order_list = [0, 1, 2, 3, 4, 5]
#     n_knots_list = [10]
#     rep_list = ['crlb', 'fim']
#     # create map
#     # gauges, links, surfaces = generate_sensors_map()
#     # plot_sensors_map(gauges, links, surfaces)
#     results_dict_fim = {}
#     results_dict_crlb = {}
#     # mcrb=MCRB()
#     for n_knots in tqdm(n_knots_list):
#         knots = np.linspace(-MAP_SHAPE[0] / 2, MAP_SHAPE[0] / 2, n_knots)
#         for order in order_list:
#             x_set = create_bsplines_set(x, order, knots)
#             y_set = create_bsplines_set(y, order, knots)
#             patch_set = create_bsplines_patch_set(x_set, y_set)
#             n_param = len(patch_set)
#             slg = signal_model.SensorGenerator(n_sensors=int(np.sqrt(len(patch_set))))
#             link_length_array = [4.0]
#             n_mc = 1
#             mc_lb_list = []
#             mc_crb_list = []
#             mc_gauge_list = []
#             for _ in tqdm(range(n_mc)):
#                 gauges = slg.generate_gauge_position(is_center_uniform=True)
#                 H_gauges = create_projection_matrix(gauges, patch_set)
#                 c_gg = np.diag(np.ones(H_gauges.shape[0])) * 0.01
#                 c_ll = np.diag(np.ones(H_gauges.shape[0])) * 0.01
#                 # print(f'{len(patch_set)=}, sensors num = {len(gauges)}')
#                 MSE_gauges_crlb = calc_bound(H_gauges, patch_set, 'crlb')
#                 trace_lb = []
#                 trace_crb = []
#                 for link_length in link_length_array:
#                     links = slg.generate_cmls(gauges=gauges, link_mean_length=link_length)
#                     H_links = create_projection_matrix(links, patch_set)
#                     theta = np.random.rand(H_links.shape[1])
#                     theta /= np.linalg.norm(theta)
#                     theta *= 24
#
#                     mcrb_obj = MCRB(H_gauges, c_gg)
#                     mcrb = mcrb_obj.compute_mcrb(c_ll)
#                     r = mcrb_obj.difference_vector(H_links, theta)
#                     lb = mcrb + np.matmul(np.reshape(r, [-1, 1]), np.reshape(r, [1, -1]))
#
#                     trace_lb.append(np.trace(lb))
#                     MSE_links_crlb = calc_bound(H_links, patch_set, 'crlb')
#                     trace_crb.append(MSE_links_crlb)
#
#                 mc_lb_list.append(trace_lb)
#                 mc_crb_list.append(trace_crb)
#                 mc_gauge_list.append(MSE_gauges_crlb)
#             results_dict_crlb[(order, n_knots)] = [np.asarray(mc_crb_list).mean(axis=0) / n_param,
#                                                    np.asarray(mc_lb_list).mean(axis=0) / n_param,
#                                                    MSE_gauges_crlb / n_param]
#             print(order, n_knots, results_dict_crlb[(order, n_knots)])
#
#             # plt.semilogy(link_length_array, results_dict_crlb[(order, n_knots)][-1] * np.ones(10), '--',
#             #              label="CRB Gauges")
#             # plt.semilogy(link_length_array, results_dict_crlb[(order, n_knots)][0], label="CRB CMLs")
#             # plt.semilogy(link_length_array, results_dict_crlb[(order, n_knots)][1],
#             #              label="MCRB True:CMLs, Assumed:Gauges")
#             # plt.legend()
#             # plt.grid()
#             # plt.show()
#
#             print("a")
#             # for n_knots in n_knots_list:
#             #     res = np.array([results_dict_fim[(order, n_knots)] for order in order_list])
#             # fig = plt.figure()
#             # plt.scatter(order_list, np.log10(res[:, 0]))
#             # plt.scatter(order_list, np.log10(res[:, 1]))
#             # plt.scatter(order_list, np.log10(res[:, 2]))
#             # plt.legend(('links', 'gauges', 'surfaces'))
#             # plt.ylabel('log10(CRLB)')
#             # plt.xlabel('Order')
#             # plt.title(f'fim reparameterization, {n_knots=}')
#             # fig.savefig(f'/Users/shay/Documents/Study/Pythonfigs/fim_reparam_{n_knots}')
#             #
#             # res = np.array([results_dict_crlb[(order, n_knots)] for order in order_list])
#             # fig = plt.figure()
#             # plt.scatter(order_list, np.log10(res[:, 0]))
#             # plt.scatter(order_list, np.log10(res[:, 1]))
#             # plt.scatter(order_list, np.log10(res[:, 2]))
#             # plt.legend(('links', 'gauges', 'surfaces'))
#             # plt.ylabel('log10(CRLB)')
#             # plt.xlabel('Order')
#             # plt.title(f'crlb reparameterization, {n_knots=}')
#             # fig.savefig(f'/Users/shay/Documents/Study/Pythonfigs/crlb_reparam_{n_knots}')
#             #
#             # plt.show(block=False)
#
#             a = 1
