

import numpy as np
import torch
from ..model_tools import ModelMethods as CPModelMethods
from ... import tensor as cp_tensor

__all__ = ['NoiseLayer']


class NoiseLayer(CPModelMethods):

    def __init__(self, scale=0.1):
        """

        :param scale: the sigma of the noise to be added to each batch input "x" is defined as the sigma of "x"
            multiplied by the "scale"
        :type scale: float | int
        """
        name_superclass = NoiseLayer.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if CPModelMethods.__name__ not in self.superclasses_initiated:
            CPModelMethods.__init__(self=self, device=None)
            if CPModelMethods.__name__ not in self.superclasses_initiated:
                self.superclasses_initiated.append(CPModelMethods.__name__)

        self.mu = 0.0
        # self.sigma = None
        self.scale = scale

        if name_superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(name_superclass)


    def forward(self, x, generator=None):
        """

        :param x: The batch input which the noise has to be added to.
        :type x: torch.Tensor | np.ndarray
        :param generator: The torch generator of the noise values.
        :type generator: torch.Generator | None
        :return: The noisy tensor.
        :rtype: torch.Tensor
        """
        # :type x: torch.Tensor # | list[torch.Tensor] | tuple[torch.Tensor]

        if self.training:
            return cp_tensor.add_noise(x=x, scale=self.scale, mu=self.mu, generator=generator)
        else:
            return x
