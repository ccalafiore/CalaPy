
import numpy as np
import torch
from ....maths import is_nan


def add_noise(x, scale=0.1, mu=0.0, generator=None):

    """This function adds gaussian noise to a tensor "x".

    :param x: The tensor which the noise has to be added to.
    :type x: torch.Tensor | np.ndarray
    :param scale: The sigma of the noise to be added to the tensor "x" is defined as the sigma of "x"
        multiplied by the "scale".
    :type scale: float | int
    :param mu: The mean of the noise.
    :type mu: float | int
    :param generator: The torch generator of the noise values.
    :type generator: torch.Generator | None
    :return: The noisy tensor.
    :rtype: torch.Tensor
    """
    # :type x: torch.Tensor # | list[torch.Tensor] | tuple[torch.Tensor]

    if isinstance(x, (torch.Tensor, np.ndarray)):

        sigma_x = torch.std(input=x, dim=None, correction=1, keepdim=False) # .item()

        # sigma_x2 = torch.std(input=x, dim=[a for a in range(0, x.ndim, 1)], correction=1, keepdim=False)
        # test = sigma_x == sigma_x2
        # max_ =torch.max(x)
        # min_ = torch.min(x)
        # diff = max_ - min_

        sigma = scale * sigma_x
        if is_nan(sigma) or (sigma == 0.0):
            # sigma = 0.0
            if mu == 0.0:
                return x
            else:
                return x + mu
        else:
            noise = generate_noise(
                shape=x.shape, mu=mu, sigma=sigma, generator=generator, dtype=x.dtype, device=x.device,
                requires_grad=False)
            return x + noise

    elif isinstance(x, (int, float)):

        if mu == 0.0:
            return x
        else:
            return x + mu
    else:
        raise TypeError('x')


def generate_noise(shape, sigma=1.0, mu=0.0, generator=None, dtype=None, device=None, requires_grad=False):

    """It generates a torch tensor with noisy values drawn from a gaussian distribution.

    :param shape: The shape of the noise tensor.
    :type shape: int | list | tuple | torch.Size | torch.Tensor | np.ndarray
    :param sigma: The standard deviation of the noise.
    :type sigma: float | int
    :param mu: The mean of the noise.
    :type mu: float | int
    :param generator: The torch generator of the noise values.
    :type generator: torch.Generator | None
    :param dtype: The torch tensor data type.
    :type dtype: torch.dtype | None
    :param device: The torch tensor device.
    :type device: torch.device | None
    :param requires_grad:
    :type requires_grad: bool
    :returns: The tensor containing the noisy values.
    :rtype: torch.Tensor
    """

    if isinstance(shape, int):
        shape = [shape]
    elif isinstance(shape, (list, tuple, torch.Size)):
        pass
    elif isinstance(shape, (torch.Tensor, np.ndarray)):
        if shape.ndim != 1:
            raise ValueError('shape')
        shape = shape.tolist()
    else:
        raise TypeError('shape')

    if sigma == 0.0:
        noise = torch.full(size=shape, fill_value=mu, dtype=dtype, device=device, requires_grad=requires_grad)
    else:

        noise = torch.randn(size=shape, generator=generator, dtype=dtype, device=device, requires_grad=requires_grad)

        if sigma != 1.0:
            noise *= sigma

        if mu != 0.0:
            noise += mu

    return noise
