

import numpy as np
import torch
from ...model_tools import ModelMethods as cp_ModelMethods
from ... import single_layers as cp_single_layers
from ._base import *
from .fcns import IndFCNNs as cp_multi_homo_layers_IndFCNNs
# from ..... import combinations as cp_combinations


__all__ = ['RNN', 'IndRNNs', 'RNNsWithSharedLayersAndPrivateLayers', 'SharedRNNAndIndRNNsAndIndFCNNs']

# todo: add non-trainable layers


class RNN(_NN):

    """A Base Class for recurrent neural networks (RNN) with homogeneous recurrent layers.

    """

    def __init__(
            self, type_name, n_features_layers, biases_layers=True, axis_time=None, h_sigma=0.1,
            nonlinearity='tanh', device=None, dtype=None):

        """

        :param type_name: The name of the type of recurrent layers. The accepted names are "rnn", "lstm" and "gru".
        :type type_name: str

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

        superclass = RNN
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

        self.accepted_type_names_with_h = tuple(['rnn', 'gru'])
        self.accepted_type_names_with_hc = tuple(['lstm'])
        self.accepted_type_names = self.accepted_type_names_with_h + self.accepted_type_names_with_hc

        if isinstance(type_name, str):
            type_name_f = type_name.lower()
            if type_name_f in self.accepted_type_names:
                self.type_name = type_name_f
            else:
                raise ValueError('type_name')
        else:
            raise TypeError('type_name')

        if self.type_name == 'rnn':

            self.layers = torch.nn.ModuleList([cp_single_layers.RNN(
                input_size=self.n_input_features_layers[l], hidden_size=self.n_output_features_layers[l],
                bias=self.biases_layers[l], nonlinearity=nonlinearity, axis_time=axis_time, h_sigma=h_sigma,
                device=device, dtype=dtype) for l in range(0, self.L, 1)])

        elif self.type_name == 'lstm':

            self.layers = torch.nn.ModuleList([cp_single_layers.LSTM(
                input_size=self.n_input_features_layers[l], hidden_size=self.n_output_features_layers[l],
                bias=self.biases_layers[l], axis_time=axis_time, h_sigma=h_sigma, device=device, dtype=dtype)
                for l in range(0, self.L, 1)])

        elif self.type_name == 'gru':

            self.layers = torch.nn.ModuleList([cp_single_layers.GRU(
                input_size=self.n_input_features_layers[l], hidden_size=self.n_output_features_layers[l],
                bias=self.biases_layers[l], axis_time=axis_time, h_sigma=h_sigma, device=device, dtype=dtype)
                for l in range(0, self.L, 1)])
        else:
            raise ValueError('type_name')

        self.axis_time, self.is_timed = self.layers[0].axis_time, self.layers[0].is_timed

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x, h=None):

        """
        :param x: A batch of the input data.
        :type x: torch.Tensor
        :type h: list[np.ndarray | torch.Tensor | None] | None
        :rtype: tuple[torch.Tensor, list[torch.Tensor]]

        """

        if h is None:
            h = [None for l in range(0, self.L, 1)]  # type: list

        if self.is_timed:
            for l in range(0, self.L, 1):
                x, h[l] = self.layers[l](x, h[l])
        elif self.type_name == 'lstm':
            for l in range(0, self.L, 1):
                h[l] = self.layers[l](x, h[l])
                x = h[l][0]
        else:
            for l in range(0, self.L, 1):
                x = h[l] = self.layers[l](x, h[l])
        return x, h

    def init_h(self, batch_shape, generators=None):

        """

        :param batch_shape: The shape of the batch input data without the time and the feature dimensions.
        :type batch_shape: int | list | tuple | torch.Size | torch.Tensor | np.ndarray
        :param generators: The instances of the torch generator to generate the tensors h with random values from a
            normal distribution.
        :type generators: list | tuple | torch.Generator | None

        :rtype: list[torch.Tensor | list[torch.Tensor]]

        """

        if generators is None:
            generators = [None for l in range(0, self.L, 1)]  # type: list
        elif isinstance(generators, torch.Generator):
            generators = [generators for l in range(0, self.L, 1)]  # type: list
        elif isinstance(generators, (list, tuple)):
            len_gens = len(generators)
            if len_gens != self.L:
                if len_gens == 0:
                    generators = [None for l in range(0, self.L, 1)]  # type: list
                elif len_gens == 1:
                    generators = [generators[0] for l in range(0, self.L, 1)]  # type: list
                else:
                    raise ValueError('len(generators)')
        else:
            raise TypeError('generators')

        # h = [None for l in range(0, self.L, 1)]  # type: list
        # for l in range(0, self.L, 1):
        #     h[l] = self.layers[l].init_h(batch_shape=batch_shape, generator=generators[l])
        # return h
        return [self.layers[l].init_h(batch_shape=batch_shape, generator=generators[l]) for l in range(0, self.L, 1)]

    def get_batch_shape_from_input_shape(self, input_shape):

        """

        :param input_shape: The input shape.
        :type input_shape: int | list | tuple | torch.Tensor | np.ndarray
        :return: The batch shape given the input shape "input_shape" and the recurrent model.
        :rtype: list[int]
        """

        return self.layers[0].get_batch_shape_from_input_shape(input_shape=input_shape)

    def set_axis_time(self, axis_time):

        """

        :type axis_time: int | None
        """

        for l in range(0, self.L, 1):

            self.layers[l].set_axis_time(axis_time=axis_time)

        self.axis_time, self.is_timed = self.layers[0].axis_time, self.layers[0].is_timed

        return self.axis_time, self.is_timed

    def concatenate_hs(self, hs, axis=0):

        n_hs = len(hs)
        # for l in range(0, self.L, 1):
        #     self.layers[l].concatenate_hs([hs[i][l] for i in range(0, n_hs, 1)], axis=axis)

        return [self.layers[l].concatenate_hs([hs[i][l] for i in range(0, n_hs, 1)], axis=axis) for l in range(0, self.L, 1)]

    def unbatch_h(self, h, axes=0, keepdims=True):

        """


        :type h: list[torch.Tensor] | list[list[torch.Tensor, torch.Tensor]]
        :type axes: list | tuple | int
        :type keepdims: bool

        """

        batch_shape = self.get_batch_shape_from_hc(h=h, axes=axes)

        hs = np.empty(shape=batch_shape + [self.L], dtype='O')
        indexes_hs = [slice(0, hs.shape[a], 1) for a in range(0, hs.ndim, 1)]  # type: list

        for l in range(0, self.L, 1):

            indexes_hs[hs.ndim - 1] = l
            hs[tuple(indexes_hs)] = self.layers[l].unbatch_h(h[l], axes=axes, keepdims=keepdims)

        return hs.tolist()

    def get_batch_shape_from_h(self, h, axes):
        return self.layers[0].get_batch_shape_from_h(h=h, axes=axes)


class IndRNNs(_IndNNs):

    """A Class of multiple independent recurrent neural networks (RNN) with homogeneous recurrent layers.


    """

    def __init__(
            self, type_name, n_features_layers, biases_layers=True, axis_features=None, axis_time=None, h_sigma=0.1,
            nonlinearity='tanh', device=None, dtype=None):

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

        superclass = IndRNNs
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

        self.accepted_type_names_with_h = tuple(['rnn', 'gru'])
        self.accepted_type_names_with_hc = tuple(['lstm'])
        self.accepted_type_names = self.accepted_type_names_with_h + self.accepted_type_names_with_hc

        if isinstance(type_name, str):
            type_name_f = type_name.lower()
            if type_name_f in self.accepted_type_names:
                self.type_name = type_name_f
            else:
                raise ValueError('type_name')
        else:
            raise TypeError('type_name')

        self.models = torch.nn.ModuleList([RNN(
            type_name=self.type_name, n_features_layers=self.n_features_layers[m], biases_layers=self.biases_layers[m],
            axis_time=axis_time, h_sigma=h_sigma, nonlinearity=nonlinearity, device=device, dtype=dtype)
            for m in range(0, self.M, 1)])

        self.axis_time, self.is_timed = self.models[0].axis_time, self.models[0].is_timed

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x, h=None):

        """

        :type x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        :type h: list[torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
        :rtype: list[list[torch.Tensor, list[torch.Tensor]]]
        """

        if isinstance(x, torch.Tensor):
            x = list(x.split(self.n_features_first_layers, dim=self.axis_features))
        elif isinstance(x, (list, tuple)):
            pass
        else:
            raise TypeError('type(x) = {}'.format(type(x)))

        if h is None:
            h = [None for m in range(0, self.M, 1)]
        elif isinstance(h, torch.Tensor):
            h = list(h.split(self.n_features_first_layers, dim=self.axis_features))
        elif isinstance(h, list):
            pass
        elif isinstance(h, tuple):
            h = list(h)
        else:
            raise TypeError('type(h) = {}'.format(type(h)))

        for m in range(0, self.M, 1):
            x[m], h[m] = self.models[m](x=x[m], h=h[m])

        return x, h

    def init_h(self, batch_shape, generators=None):

        """

        :param batch_shape: The shape of the batch input data without the time and the feature dimensions.
        :type batch_shape: int | list | tuple | torch.Size | torch.Tensor | np.ndarray
        :param generators: The instances of the torch generator to generate the tensors h with random values from a
            normal distribution.
        :type generators: list | tuple | torch.Generator | None

        :rtype: list[list[torch.Tensor | list[torch.Tensor]]]

        """

        if generators is None:
            generators = [None for m in range(0, self.M, 1)]  # type: list
        elif isinstance(generators, torch.Generator):
            generators = [generators for m in range(0, self.M, 1)]  # type: list
        elif isinstance(generators, (list, tuple)):
            len_gens = len(generators)
            if len_gens != self.M:
                if len_gens == 0:
                    generators = [None for m in range(0, self.M, 1)]  # type: list
                elif len_gens == 1:
                    generators = [generators[0] for m in range(0, self.M, 1)]  # type: list
                else:
                    raise ValueError('len(generators)')
        else:
            raise TypeError('generators')

        # h = [None for m in range(0, self.M, 1)]  # type: list
        # for m in range(0, self.M, 1):
        #     h[m] = self.models[m].init_h(batch_shape=batch_shape, generator=generators[m])
        # return h
        return [self.models[m].init_h(batch_shape=batch_shape, generators=generators[m]) for m in range(0, self.M, 1)]

    def get_batch_shape_from_input_shape(self, input_shape):

        """

        :param input_shape: The input shape.
        :type input_shape: int | list | tuple | torch.Tensor | np.ndarray
        :return: The batch shape given the input shape "input_shape" and the recurrent model.
        :rtype: list[int]
        """

        return self.models[0].get_batch_shape_from_input_shape(input_shape=input_shape)

    def set_axis_time(self, axis_time):

        """

        :type axis_time: int | None
        """

        for m in range(0, self.M, 1):

            self.models[m].set_axis_time(axis_time=axis_time)

        self.axis_time, self.is_timed = self.models[0].axis_time, self.models[0].is_timed

        return self.axis_time, self.is_timed

    def concatenate_hs(self, hs, axis=0):

        n_hs = len(hs)
        return [self.models[m].concatenate_hs([hs[i][m] for i in range(0, n_hs, 1)], axis=axis) for m in range(0, self.M, 1)]

    def unbatch_h(self, h, axes=0, keepdims=True):

        """


        :type h: list[list[torch.Tensor]] | list[list[list[torch.Tensor, torch.Tensor]]]
        :type axes: list | tuple | int
        :type keepdims: bool

        """

        batch_shape = self.get_batch_shape_from_hc(h=h, axes=axes)

        hs = np.empty(shape=batch_shape + [self.M], dtype='O')
        indexes_hs = [slice(0, hs.shape[a], 1) for a in range(0, hs.ndim, 1)]  # type: list

        for m in range(0, self.M, 1):
            indexes_hs[hs.ndim - 1] = m
            hs[tuple(indexes_hs)] = self.models[m].unbatch_h(h[m], axes=axes, keepdims=keepdims)

        return hs.tolist()

    def get_batch_shape_from_h(self, h, axes):
        return self.models[0].get_batch_shape_from_h(h=h, axes=axes)


class RNNsWithSharedLayersAndPrivateLayers(cp_ModelMethods):

    def __init__(
            self, type_name, n_features_shared_layers, n_features_private_layers,
            biases_shared_layers=True, biases_private_layers=True,
            axis_features=None, axis_time=None, h_sigma=0.1, nonlinearity='tanh',
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

        superclass = RNNsWithSharedLayersAndPrivateLayers
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

        self.accepted_type_names_with_h = tuple(['rnn', 'gru'])
        self.accepted_type_names_with_hc = tuple(['lstm'])
        self.accepted_type_names = self.accepted_type_names_with_h + self.accepted_type_names_with_hc

        if isinstance(type_name, str):
            type_name_f = type_name.lower()
            if type_name_f in self.accepted_type_names:
                self.type_name = type_name_f
            else:
                raise ValueError('type_name')
        else:
            raise TypeError('type_name')

        self.shared_layers = RNN(
            type_name=self.type_name, n_features_layers=n_features_shared_layers, biases_layers=biases_shared_layers,
            axis_time=axis_time, h_sigma=h_sigma, nonlinearity=nonlinearity, device=device, dtype=dtype)

        self.private_layers = IndRNNs(
            type_name=self.type_name, n_features_layers=n_features_private_layers, biases_layers=biases_private_layers,
            axis_features=axis_features, axis_time=axis_time, h_sigma=h_sigma, nonlinearity=nonlinearity,
            device=device, dtype=dtype)

        self.axis_time, self.is_timed = self.shared_layers.axis_time, self.shared_layers.is_timed

        if self.shared_layers.n_features_layers[-1] != self.private_layers.n_features_first_layers_together:
            raise ValueError('n_features_shared_layers[-1], n_features_private_layers[:][0]')

        self.M = self.private_layers.M

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x, h=None):

        """

        :type x: torch.Tensor
        :type h: torch.Tensor
        :rtype: list[list[torch.Tensor, list[torch.Tensor]]]
        """
        if h is None:
            h = [None, None]

        x, h[0] = self.shared_layers(x=x, h=h[0])
        x, h[1] = self.private_layers(x=x, h=h[1])
        return x, h

    def init_h(self, batch_shape, generators=None):

        """

        :param batch_shape: The shape of the batch input data without the time and the feature dimensions.
        :type batch_shape: int | list | tuple | torch.Size | torch.Tensor | np.ndarray
        :param generators: The instances of the torch generator to generate the tensors h with random values from a
            normal distribution.
        :type generators: list | tuple | torch.Generator | None

        :rtype: list[list[torch.Tensor | list[torch.Tensor]]]

        """

        if generators is None:
            generators = [None, None]  # type: list
        elif isinstance(generators, torch.Generator):
            generators = [generators, generators]  # type: list
        elif isinstance(generators, (list, tuple)):
            len_gens = len(generators)
            if len_gens != 2:
                if len_gens == 0:
                    generators = [None, None]  # type: list
                elif len_gens == 1:
                    generators = [generators[0], generators[0]]  # type: list
                else:
                    raise ValueError('len(generators)')
        else:
            raise TypeError('generators')

        # h = [None for m in range(0, self.M, 1)]  # type: list
        # for m in range(0, self.M, 1):
        #     h[m] = self.models[m].init_h(batch_shape=batch_shape, generator=generators[m])
        # return h
        return [self.shared_layers.init_h(batch_shape=batch_shape, generators=generators[0]),
                self.private_layers.init_h(batch_shape=batch_shape, generators=generators[1])]

    def get_batch_shape_from_input_shape(self, input_shape):

        """

        :param input_shape: The input shape.
        :type input_shape: int | list | tuple | torch.Tensor | np.ndarray
        :return: The batch shape given the input shape "input_shape" and the recurrent model.
        :rtype: list[int]
        """

        return self.shared_layers.get_batch_shape_from_input_shape(input_shape=input_shape)

    def set_axis_time(self, axis_time):

        """

        :type axis_time: int | None
        """
        self.shared_layers.set_axis_time(axis_time=axis_time)

        self.private_layers.set_axis_time(axis_time=axis_time)

        self.axis_time, self.is_timed = self.shared_layers.axis_time, self.shared_layers.is_timed

        return self.axis_time, self.is_timed

    def concatenate_hs(self, hs, axis=0):

        n_hs = len(hs)
        return [
            self.shared_layers.concatenate_hs([hs[i][0] for i in range(0, n_hs, 1)], axis=axis),
            self.private_layers.concatenate_hs([hs[i][1] for i in range(0, n_hs, 1)], axis=axis)]

    def unbatch_h(self, h, axes=0, keepdims=True):

        """


        :type h: list[list[torch.Tensor], list[list[torch.Tensor]]] | list[list[list[torch.Tensor, torch.Tensor]], list[list[list[torch.Tensor, torch.Tensor]]]]
        :type axes: list | tuple | int
        :type keepdims: bool

        """
        n = 2
        batch_shape = self.get_batch_shape_from_hc(h=h, axes=axes)

        hs = np.empty(shape=batch_shape + [n], dtype='O')
        indexes_hs = [slice(0, hs.shape[a], 1) for a in range(0, hs.ndim, 1)]  # type: list

        i = 0
        indexes_hs[hs.ndim - 1] = i
        hs[tuple(indexes_hs)] = self.shared_layers.unbatch_h(h[i], axes=axes, keepdims=keepdims)
        i += 1
        indexes_hs[hs.ndim - 1] = i
        hs[tuple(indexes_hs)] = self.shared_layers.unbatch_h(h[i], axes=axes, keepdims=keepdims)

        return hs.tolist()

    def get_batch_shape_from_h(self, h, axes):
        return self.shared_layers[0].get_batch_shape_from_h(h=h, axes=axes)


class SharedRNNAndIndRNNsAndIndFCNNs(RNNsWithSharedLayersAndPrivateLayers):

    def __init__(
            self, type_name, n_features_shared_rnn_layers, n_features_private_rnn_layers, n_features_private_fc_layers,
            biases_shared_rnn_layers=True, biases_private_rnn_layers=True, biases_private_fc_layers=True,
            axis_features=None, axis_time=None, h_sigma=0.1, nonlinearity='tanh',
            device=None, dtype=None):

        """

        :param n_features_shared_rnn_layers: A sequence of ints whose first element is the input size of the first shared
            layer and all other elements from the second to the last are the output sizes of all shared layers. The
            models share L layers, where L=len(n_features_shared_rnn_layers)-1. The n_features_shared_rnn_layers must have 2
            elements or more such that the NN can have 1 shared layer or more.
        :type n_features_shared_rnn_layers: list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]

        :param n_features_private_rnn_layers: A sequence of M sequences, one per NN, where M is the number of NNs with
            private layers. The M sequences can have different sizes, but each size should be 2 or more. The first
            element of the m_th sequence is the input size of the first private layer of m_th NN. The sum of the input
            sizes of the first private layers of all models must be equal to the output size of the last shared layer of
            all models (i.e. sum(n_features_private_rnn_layers[0][0], n_features_private_rnn_layers[1][0], ...,
            n_features_private_rnn_layers[M-1][0]) = n_features_shared_rnn_layers[-1]). The other elements from the second to
            the last of the m_th sequence are the output sizes (the actual layer sizes) of all private layers of the
            m_th NN. The m_th NN have L[m] private layers, where L[m]=len(n_features_private_rnn_layers[m])-1.
        :type n_features_private_rnn_layers: list[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]] |
                                 tuple[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]]

        :param biases_shared_rnn_layers: A bool or sequence of L bools where L is the number of shared layers of all model.
            If it is a bool and is True (default), all shared layers of all models WILL HAVE trainable biases. If it is
            a bool and is False, all shared layers of all models WILL NOT HAVE trainable biases. If it is a sequence of
            L bools and biases_rnn_layers[l] is True, then the l_th shared layer of the NNs WILL HAVE trainable biases. If
            it is a sequence of L bools and biases_rnn_layers[l] is False, then the l_th shared layer of the models WILL NOT
            HAVE trainable biases.
        :type biases_shared_rnn_layers: bool | list[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]]

        :param biases_private_rnn_layers: A bool, sequence of M bools, or a sequence of M sequences of L[m] bools where M is
            the number of models with private deeper layers and L[m] is the number of private layers of the
            m_th model. If it is a bool and is True (default), all private layers of all models WILL HAVE trainable
            biases. If it is a bool and is False, all private layers of all models WILL NOT HAVE trainable biases.
            If it is a sequence of M bools and biases_rnn_layers[m] is True, then all private layers of the m_th NN WILL
            HAVE trainable biases. If it is a sequence of M bools and biases_rnn_layers[m] is False, then all private
            layers of the m_th model WILL NOT HAVE trainable biases. If it is a sequence of M sequences of L[m] bools
            and biases_rnn_layers[m][l] is True, then the l_th private layer of the m_th model WILL HAVE trainable biases.
            If it is a sequence of M sequences of L[m] bools and biases_rnn_layers[m][l] is False, then the l_th private
            layer of the m_th model WILL NOT HAVE trainable biases.
        :type biases_private_rnn_layers: bool |
            list[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]] |
            tuple[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]] |
            np.ndarray[bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]]

        :param device: The torch device of all layers.
        :type device: torch.device | str | int | None

        :param dtype: The torch data type of all layers.
        :type dtype: torch.dtype | str| None
        """

        superclass = SharedRNNAndIndRNNsAndIndFCNNs
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if RNNsWithSharedLayersAndPrivateLayers not in self.superclasses_initiated:
            RNNsWithSharedLayersAndPrivateLayers.__init__(
                self=self, type_name=type_name,
                n_features_shared_layers=n_features_shared_rnn_layers,
                n_features_private_layers=n_features_private_rnn_layers,
                biases_shared_layers=biases_shared_rnn_layers, biases_private_layers=biases_private_rnn_layers,
                axis_features=axis_features, axis_time=axis_time, h_sigma=h_sigma, nonlinearity=nonlinearity,
                device=device, dtype=dtype)

            if RNNsWithSharedLayersAndPrivateLayers not in self.superclasses_initiated:
                self.superclasses_initiated.append(RNNsWithSharedLayersAndPrivateLayers)

        self.private_fc_layers = cp_multi_homo_layers_IndFCNNs(
            n_features_layers=n_features_private_fc_layers, biases_layers=biases_private_fc_layers,
            axis_features=axis_features, device=device, dtype=dtype)

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x, h=None):

        """

        :type x: torch.Tensor
        :type h: torch.Tensor
        :rtype: list[list[torch.Tensor, list[torch.Tensor]]]
        """
        if h is None:
            h = [None, None]

        x, h[0] = self.shared_layers(x=x, h=h[0])
        x, h[1] = self.private_layers(x=x, h=h[1])

        return self.private_fc_layers(x=x), h
