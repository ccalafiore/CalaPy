

import numpy as np
import torch
from ..model_tools import ModelMethods as CPModelMethods
from ... import tensors as cp_tensors

__all__ = ['NoiseLayer']


class NoiseLayer(CPModelMethods):

    def __init__(self, scale=0.1):
        """

        :param scale: the sigma of the noise to be added to each batch input "x" is defined as the sigma of "x"
            multiplied by the "scale"
        :type scale: float | int
        """
        superclass = NoiseLayer
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if CPModelMethods not in self.superclasses_initiated:
            CPModelMethods.__init__(self=self)
            if CPModelMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(CPModelMethods)

        self.mu = 0.0
        # self.sigma = None
        self.scale = scale

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

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
            return cp_tensors.add_noise(x=x, scale=self.scale, mu=self.mu, generator=generator)
        else:
            return x
