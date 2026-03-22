import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt


def gamma_pdf(in_x, in_alpha, in_beta):
    g_alpha = gamma(in_alpha)
    scale = np.power(in_beta, in_alpha) / g_alpha
    return np.power(in_x, in_alpha - 1) * np.exp(-in_beta * in_x) * scale


x = np.linspace(0.001, 40, 1000)

# plt.plot(x, gamma_pdf(x, 1, 2))
# plt.plot(x, gamma_pdf(x, 20, 2))
plt.plot(x, gamma_pdf(x, 3, 2))
plt.plot(x, gamma_pdf(x,5, 2))
plt.plot(x, gamma_pdf(x,12, 2))
plt.plot(x, gamma_pdf(x,30, 2))
plt.plot(x, gamma_pdf(x,30, 1.2))
plt.show()
# x = np.linspace(gamma.ppf(0.01, a),
#                 gamma.ppf(0.99, a), 100)
