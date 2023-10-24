

import numpy as np
import torch
from ...model_tools import ModelMethods as cp_ModelMethods
from ._base import *
# from ..... import combinations as cp_combinations


__all__ = ['FCNN', 'IndFCNNs', 'FCNNsWithSharedLayersAndPrivateLayers']

# todo: add non-trainable layers


class FCNN(_NN):

    """A Class of neural networks (NN) with only fully-connected (FC) layers.


    """

    def __init__(self, n_features_layers, biases_layers=True, device=None, dtype=None):

        """

        :param n_features_layers: A sequence of ints whose first element is the input size of the NN and all other
            elements from the second to the last are the output sizes of all layers. The NN will have L layers, where
            L=len(n_features_layers)-1. The n_features_layers must have 2 elements or more such that the NN can have 1
            layer or more.
        :type n_features_layers: list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]

        :param biases_layers: A bool or a sequence of bools with L elements where L is the number of trainable layers.
            If it is a bool and is True (default), all layers WILL HAVE trainable biases. If it is a bool and is False,
            all layers WILL NOT HAVE trainable biases. If it is a 1d vector of L bools and biases_layers[l] is
            True, the l_th layer WILL HAVE trainable biases. If it is a 1d vector of L bools and biases_layers[l] is
            False, then the l_th layer WILL NOT HAVE trainable biases.
        :type biases_layers: bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]

        :param device: The torch device of the layers.
        :type device: torch.device | str | int | None

        :param dtype: The torch data type of the layers.
        :type dtype: torch.dtype | str| None
        """

        superclass = FCNN
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if _NN not in self.superclasses_initiated:
            _NN.__init__(self=self, n_features_layers=n_features_layers, biases_layers=biases_layers)
            if _NN not in self.superclasses_initiated:
                self.superclasses_initiated.append(_NN)

        self.layers = torch.nn.Sequential(*[torch.nn.Linear(
            in_features=self.n_input_features_layers[l], out_features=self.n_output_features_layers[l],
            bias=self.biases_layers[l], device=device, dtype=dtype) for l in range(0, self.L, 1)])

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x):

        """

        :param x: A batch of the input data.
        :type x: torch.Tensor | np.ndarray
        :rtype: torch.Tensor
        """

        return self.layers(x)


class IndFCNNs(_IndNNs):

    """A Class of multiple independent neural networks (NN) with only fully-connected (FC) layers.


    """

    def __init__(self, n_features_layers, biases_layers=True, axis_features=None, device=None, dtype=None):

        """

        :param n_features_layers: A sequence of M sequences, one per neural network (NN), where M is the number of
            independent NNs. The M sequences can have different sizes, but each size should be 2 or more. The first
            element of the m_th sequence is the input size of the m_th NN, while the other elements from the second to
            the last of the m_th sequence are the output sizes (the actual layer sizes) of all layers of the m_th NN.
            The m_th NN will have L[m] layers, where L[m]=len(n_features_layers[m])-1.
        :type n_features_layers: list[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]] |
                                 tuple[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]]

        :param biases_layers: A bool, sequence of M bools, or a sequence of M sequences of L[m] bools where M is
            the number of independent models (or NNs) with private deeper layers and L[m] is the number of private
            layers of the m_th independent model. If it is a bool and is True (default), all layers WILL HAVE trainable
            biases. If it is a bool and is False, all layers WILL NOT HAVE trainable biases. If it is a sequence of M
            bools and biases_layers[m] is True, then all layers of the m_th NN WILL HAVE trainable biases. If it is a
            sequence of M bools and biases_layers[m] is False, then all layers of the m_th NN WILL NOT HAVE trainable
            biases. If it is a sequence of M sequences of L[m] bools and biases_layers[m][l] is True, then the l_th
            layer of the m_th model WILL HAVE trainable biases. If it is a sequence of M sequences of L[m] bools and
            biases_layers[m][l] is False, then the l_th layer of the m_th model WILL NOT HAVE trainable biases.
        :type biases_layers: bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]

        :param device: The torch device of the layers.
        :type device: torch.device | str | int | None

        :param dtype: The torch data type of the layers.
        :type dtype: torch.dtype | str| None
        """

        superclass = IndFCNNs
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if _IndNNs not in self.superclasses_initiated:
            _IndNNs.__init__(
                self=self, n_features_layers=n_features_layers, biases_layers=biases_layers,
                axis_features=axis_features)
            if _IndNNs not in self.superclasses_initiated:
                self.superclasses_initiated.append(_IndNNs)

        self.layers = torch.nn.ModuleList([FCNN(
            n_features_layers=self.n_features_layers[m], biases_layers=self.biases_layers[m], device=device,
            dtype=dtype) for m in range(0, self.M, 1)])

        # self.layers = torch.nn.ModuleList([torch.nn.Sequential(
        #     *[torch.nn.Linear(
        #         in_features=self.n_input_features_layers[m][l], out_features=self.n_output_features_layers[m][l],
        #         bias=self.biases_layers[m][l], device=device, dtype=dtype)
        #         for l in range(1, self.L[m], 1)]) for m in range(0, self.M, 1)])

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x):

        """

        :type x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        :rtype: list[torch.Tensor]
        """

        if isinstance(x, torch.Tensor):
            x = x.split(self.n_features_first_layers, dim=self.axis_features)
        elif isinstance(x, (list, tuple)):
            pass
        else:
            raise TypeError('type(x) = {}'.format(type(x)))

        return [self.layers[m](x[m]) for m in range(0, self.M, 1)]


class FCNNsWithSharedLayersAndPrivateLayers(cp_ModelMethods):

    def __init__(
            self, n_features_shared_layers, n_features_private_layers,
            biases_shared_layers=True, biases_private_layers=True, axis_features=None,
            device=None, dtype=None):

        """

        :param n_features_shared_layers: A sequence of ints whose first element is the input size of the first shared
            layer and all other elements from the second to the last are the output sizes of all shared layers. The
            models share L layers, where L=len(n_features_shared_layers)-1. The n_features_shared_layers must have 2
            elements or more such that the NN can have 1 shared layer or more.
        :type n_features_shared_layers: list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]

        :param n_features_private_layers: A sequence of M sequences, one per NN, where M is the number of NNs with
            private layers. The M sequences can have different sizes, but each size should be 2 or more. The first
            element of the m_th sequence is the input size of the first private layer of m_th NN. The sum of the input
            sizes of the first private layers of all models must be equal to the output size of the last shared layer of
            all models (i.e. sum(n_features_private_layers[0][0], n_features_private_layers[1][0], ...,
            n_features_private_layers[M-1][0]) = n_features_shared_layers[-1]). The other elements from the second to
            the last of the m_th sequence are the output sizes (the actual layer sizes) of all private layers of the
            m_th NN. The m_th NN have L[m] private layers, where L[m]=len(n_features_private_layers[m])-1.
        :type n_features_private_layers: list[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]] |
                                 tuple[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]]

        :param biases_shared_layers: A bool or sequence of L bools where L is the number of shared layers of all model.
            If it is a bool and is True (default), all shared layers of all models WILL HAVE trainable biases. If it is
            a bool and is False, all shared layers of all models WILL NOT HAVE trainable biases. If it is a sequence of
            L bools and biases_layers[l] is True, then the l_th shared layer of the NNs WILL HAVE trainable biases. If
            it is a sequence of L bools and biases_layers[l] is False, then the l_th shared layer of the models WILL NOT
            HAVE trainable biases.
        :type biases_shared_layers: bool | list[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]]

        :param biases_private_layers: A bool, sequence of M bools, or a sequence of M sequences of L[m] bools where M is
            the number of models with private deeper layers and L[m] is the number of private layers of the
            m_th model. If it is a bool and is True (default), all private layers of all models WILL HAVE trainable
            biases. If it is a bool and is False, all private layers of all models WILL NOT HAVE trainable biases.
            If it is a sequence of M bools and biases_layers[m] is True, then all private layers of the m_th NN WILL
            HAVE trainable biases. If it is a sequence of M bools and biases_layers[m] is False, then all private
            layers of the m_th model WILL NOT HAVE trainable biases. If it is a sequence of M sequences of L[m] bools
            and biases_layers[m][l] is True, then the l_th private layer of the m_th model WILL HAVE trainable biases.
            If it is a sequence of M sequences of L[m] bools and biases_layers[m][l] is False, then the l_th private
            layer of the m_th model WILL NOT HAVE trainable biases.
        :type biases_private_layers: bool |
            list[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]] |
            tuple[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]] |
            np.ndarray[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]]

        :param device: The torch device of all layers.
        :type device: torch.device | str | int | None

        :param dtype: The torch data type of all layers.
        :type dtype: torch.dtype | str| None
        """

        superclass = FCNNsWithSharedLayersAndPrivateLayers
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if cp_ModelMethods not in self.superclasses_initiated:
            cp_ModelMethods.__init__(self=self)
            if cp_ModelMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(cp_ModelMethods)

        self.shared_layers = FCNN(
            n_features_layers=n_features_shared_layers,
            biases_layers=biases_shared_layers, device=device, dtype=dtype)

        self.private_layers = IndFCNNs(
            n_features_layers=n_features_private_layers,
            biases_layers=biases_private_layers, axis_features=axis_features, device=device, dtype=dtype)

        if self.shared_layers.n_features_layers[-1] != self.private_layers.n_features_first_layers_together:
            raise ValueError('n_features_shared_layers[-1], n_features_private_layers[:][0]')

        self.M = self.private_layers.M

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x):

        """

        :type x: torch.Tensor
        :rtype: list[torch.Tensor]
        """

        return self.private_layers(self.shared_layers(x))
