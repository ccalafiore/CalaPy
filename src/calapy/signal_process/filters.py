
import numpy as np
import math


def gauss_nd(x, mu=0, sigma=1, a=None):


    """

    https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    :param x: n-dimensional point
    :type x: list[int | float] | tuple[int | float] | int | float
    :param mu: The center of the peak of the "bell"
    :type mu: list[int | float] | tuple[int | float] | int | float | None
    :param sigma: The width of the "bell"
    :type sigma: list[int | float] | tuple[int | float] | int | float | None
    :param a: The height of the curve's peak
    :type a: int | float | None
    :return:
    """

    x = format_floats(x=x)
    mu = format_floats(x=mu)
    sigma = format_floats(x=sigma)

    n_dims = len(x)

    len_mu = len(mu)
    if len_mu == n_dims:
        pass
    elif len_mu == 1:
        mu = [mu[0] for _ in range(0, n_dims, 1)]
    else:
        raise ValueError('mu')

    len_sigma = len(sigma)
    if len_sigma == n_dims:
        pass
    elif len_sigma == 1:
        sigma = [sigma[0] for _ in range(0, n_dims, 1)]
    else:
        raise ValueError('sigma')

    if a is None:
        two_pis = 2.0 * math.pi
        a = 1.0 / prod([sigma[d] * math.sqrt(two_pis) for d in range(0, n_dims, 1)])

    a = format_float(x=a)

    e = np.exp(-sum([((x[d] - mu[d])**2.0) / (2.0 * (sigma[d]**2.0)) for d in range(0, n_dims, 1)]))

    if a == 1.0:
        return e
    else:
        return a * e


def gauss_1d(x, mu=0, sigma=1, a=None):

    x = format_float(x=x)
    mu = format_float(x=mu)
    sigma = format_float(x=sigma)

    if a is None:
        a = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    return gauss_nd(x=[x, y], mu=[mu, mu], sigma=[sigma, sigma], a=a)


def gauss_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, a=None):

    return gauss_nd(x=[x, y], mu=[mu_x, mu_y], sigma=[sigma_x, sigma_y], a=a)


def gauss_3d(x, y, z, mu_x=0, mu_y=0, mu_z=0, sigma_x=1, sigma_y=1, sigma_z=1, a=None):

    return gauss_nd(x=[x, y, z], mu=[mu_x, mu_y, mu_z], sigma=[sigma_x, sigma_y, sigma_z], a=a)


def gauss_kernel_nd(n_dims, sigma=3, size=9, mu=0):

    if not isinstance(n_dims, int):
        raise TypeError('n_dims')

    size = format_ints(x=size)
    len_size = len(size)
    if len_size == n_dims:
        pass
    elif len_size == 1:
        size = [size[0] for _ in range(0, n_dims, 1)]
    else:
        raise ValueError('size')

    half_size = [math.floor(size[d] / 2) for d in range(0, n_dims, 1)]

    starts = [-half_size[d] for d in range(0, n_dims, 1)]

    stops = [half_size[d] if ((size[d] % 2) == 0) else (half_size[d] + 1) for d in range(0, n_dims, 1)]

    grids = np.meshgrid(*[np.arange(starts[d], stops[d], 1) for d in range(0, n_dims, 1)], indexing='ij')

    kernel = gauss_nd(x=grids, mu=mu, sigma=sigma, a=1.0)

    kernel = (1 / np.sum(a=kernel, axis=None)) * kernel

    return kernel
