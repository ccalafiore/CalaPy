
import random
import numpy as np
import torch


class Noise:


    def __init__(self, mu=0.0, sigma=1.0):

        self.mu = mu
        self.sigma = sigma


    def forward(self, x, generator=None):

        """

        :type x: torch.Tensor

        :type generator: torch.Generator | None
        :rtype: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        """

        # :type x: torch.Tensor # | list[torch.Tensor] | tuple[torch.Tensor]

        if isinstance(x, (torch.Tensor, np.ndarray)):

            if self.sigma != 0.0:

                noise = torch.randn(
                    size=x.shape, generator=generator, dtype=x.dtype, device=x.device, requires_grad=False)

                if self.sigma != 1.0:
                    noise *= self.sigma

                x += noise

            if self.mu == 0.0:
                return x
            else:
                return x + self.mu

        # elif isinstance(x, (tuple, list)):
        #     type_x = type(x)
        #     x = [self(x[i], generator=generator) for i in range(0, len(x), 1)]
        #     if type_x == tuple:
        #         x = tuple(x)

        elif isinstance(x, (int, float)):

            if self.sigma == 0.0:
                if self.mu == 0.0:
                    return x
                else:
                    return x + self.mu
            else:
                noise = random.gauss(mu=self.mu, sigma=self.sigma)
                x += noise
                return x

        else:
            raise TypeError('x')

    __call__ = forward
