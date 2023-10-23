
"""The module for heterogeneous neural networks (NNs).

The heterogeneous NNs can have heterogeneous layers (i.e. some fully-connected, some recurrent, some convolutional, and
so on).
"""

import typing
import numpy as np
import torch
from .homo import FCNN, IndFCNNs, LSTMNNs
from ..model_tools import ModelMethods as cp_ModelMethods
from .. import single_layers as cp_single_layers
# from ... import tensors as cp_tensors

__all__ = ['SequentialHeteroLayers']


class SequentialHeteroLayers(cp_ModelMethods):

    """The class of basic neural networks (NNs)

    """

    def __init__(self, params_of_layers, device=None, dtype=None):

        """

        :param params_of_layers:
        :type params_of_layers: list[dict] | tuple[dict] | np.ndarray[dict]
        :type device: torch.device | str | int | None
        :type dtype: torch.dtype | str| None

        """

        superclass = SequentialHeteroLayers
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

        if isinstance(params_of_layers, (list, tuple, np.ndarray)):
            if isinstance(params_of_layers, list):
                tmp_params_of_layers = tuple(params_of_layers)
            elif isinstance(params_of_layers, tuple):
                tmp_params_of_layers = params_of_layers
            elif isinstance(params_of_layers, (np.ndarray, torch.Tensor)):
                tmp_params_of_layers = tuple(params_of_layers.tolist())
            else:
                raise TypeError('params_of_layers')

        else:
            raise TypeError('params_of_layers')

        self.L = self.n_layers = len(tmp_params_of_layers)

        self.accepted_layer_types_with_trainable_params = tuple(
            ['fc', 'rnn', 'lstm', 'gru', 'conv1d',  'conv2d', 'conv3d'])
        self.accepted_layer_types_without_trainable_params = tuple(
            ['noise', 'addition', 'concatenation', 'dropout', 'sigmoid', 'tanh', 'relu', 'flatten'])
        self.all_accepted_layer_types = (
                self.accepted_layer_types_with_trainable_params + self.accepted_layer_types_without_trainable_params)

        for l in range(0, self.L, 1):
            if isinstance(tmp_params_of_layers[l]['type_name'], str):
                lower_layer_type_l = tmp_params_of_layers[l]['type_name'].lower()
                if lower_layer_type_l in self.all_accepted_layer_types:
                    tmp_params_of_layers[l]['type_name'] = lower_layer_type_l
                else:
                    raise ValueError(f'params_of_layers[{l:d}][\'type_name\']')
            else:
                raise TypeError(f'params_of_layers[{l:d}][\'type_name\']')

        self.params_of_layers = tmp_params_of_layers
        # self.layer_types = tuple([self.params_of_layers[l]['type_name'] for l in range(0, self.L, 1)])

        # define self.layers
        self.layers = torch.nn.ModuleList()
        self.hidden_state_sizes = []
        self.recurrent_layer_indexes = []
        self.recurrent_layer_types = []

        for l in range(0, self.L, 1):
            if self.params_of_layers[l]['type_name'] == 'fc':
                layer_l = torch.nn.Linear(**self.params_of_layers[l]['params'], device=device, dtype=dtype)
            elif self.params_of_layers[l]['type_name'] == 'rnn':
                layer_l = cp_single_layers.RNN(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'lstm':
                layer_l = cp_single_layers.LSTM(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'gru':
                layer_l = cp_single_layers.GRU(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'conv1d':
                layer_l = cp_single_layers.Conv1d(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
            elif self.params_of_layers[l]['type_name'] == 'conv2d':
                layer_l = cp_single_layers.Conv2d(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
            elif self.params_of_layers[l]['type_name'] == 'conv3d':
                layer_l = cp_single_layers.Conv3d(
                    **self.params_of_layers[l]['params'], device=device, dtype=dtype)
            elif self.params_of_layers[l]['type_name'] == 'noise':
                layer_l = cp_single_layers.Noise(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'addition':
                layer_l = cp_single_layers.Addition(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'concatenation':
                layer_l = cp_single_layers.Concatenation(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'dropout':
                layer_l = torch.nn.Dropout(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'sigmoid':
                layer_l = torch.nn.Sigmoid(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'tanh':
                layer_l = torch.nn.Tanh(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'relu':
                layer_l = torch.nn.ReLU(**self.params_of_layers[l]['params'])
            elif self.params_of_layers[l]['type_name'] == 'flatten':
                layer_l = torch.nn.Flatten(**self.params_of_layers[l]['params'])

            # todo: add extra layer types.
            # you can add here other types of layers with an extra elif statement for each extra layer type

            else:
                raise ValueError(f'params_of_layers[{l:d}][\'type_name\']')

            self.layers.append(module=layer_l)

        if superclass == type(self):
            self.get_device()
            self.get_dtype()

        self.Z = self.n_recurrent_layers = len(self.hidden_state_sizes)
        self.is_with_any_recurrent_layers = self.Z > 0

        if self.is_with_any_recurrent_layers:
            self.forward = self._forward_with_recurrent_layers
        else:
            self.forward = self._forward_without_recurrent_layers

    def _forward_without_recurrent_layers(self, x):

        """

        :param x: A batch of the input data.
        :type x: torch.Tensor | np.ndarray
        """

        for l in range(0, self.L, 1):
            x = self.layers[l](x)

        return x

    def _forward_with_recurrent_layers(self, x, h=None):

        """
        :param x: A batch of the input data.
        :type x: torch.Tensor | np.ndarray
        :type h: list[list[np.ndarray | torch.Tensor | None] |
                 tuple[np.ndarray | torch.Tensor | None] |
                 np.ndarray | torch.Tensor | None] | None
        :rtype: tuple[torch.Tensor, list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]]


        """

        if h is None:
            h = [None for z in range(0, self.Z, 1)]  # type: list

        z = 0

        for l in range(0, self.L, 1):

            if self.params_of_layers[l]['type_name'] in ['rnn', 'lstm', 'gru']:

                x, h[z] = self.layers[l](x, h[z])

                z += 1
            else:
                x = self.layers[l](x)

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
            generators = [None for z in range(0, self.Z, 1)]  # type: list
        elif isinstance(generators, torch.Generator):
            generators = [generators for z in range(0, self.Z, 1)]  # type: list
        elif isinstance(generators, (list, tuple)):
            len_gens = len(generators)
            if len_gens != self.Z:
                if len_gens == 0:
                    generators = [None for z in range(0, self.Z, 1)]  # type: list
                elif len_gens == 1:
                    generators = [generators[0] for z in range(0, self.Z, 1)]  # type: list
                else:
                    raise ValueError('len(generators)')
        else:
            raise TypeError('generators')

        h = [None for z in range(0, self.Z, 1)]  # type: list

        for z in range(0, self.Z, 1):

            h[z] = self.layers[self.recurrent_layer_indexes[z]].init_h(batch_shape=batch_shape, generator=generators[z])

        return h

    def get_batch_shape(self, input_shape, batch_axes):

        """

        :param input_shape: The input shape.
        :type input_shape: int | list | tuple | torch.Tensor | np.ndarray
        :param batch_axes: The batch axes of the input.
        :type input_shape: int | list | tuple | slice | torch.Tensor | np.ndarray
        :return: The batch shape given the input shape "input_shape" and the batch axes "batch_axes".
        :rtype: list[int]
        """

        if isinstance(input_shape, int):
            input_shape_f = np.asarray(a=[input_shape], dtype='i')
        elif isinstance(input_shape, (list, tuple)):
            input_shape_f = np.asarray(a=input_shape, dtype='i')
        elif isinstance(input_shape, (torch.Tensor, np.ndarray)):
            input_shape_f = input_shape
        else:
            raise TypeError('input_shape')

        if isinstance(batch_axes, int):
            batch_axes_f = [batch_axes]
        elif isinstance(batch_axes, (list, tuple, slice, torch.Tensor, np.ndarray)):
            batch_axes_f = batch_axes
        else:
            raise TypeError('batch_axes')

        batch_shape = input_shape_f[batch_axes_f].tolist()

        return batch_shape


class SequentialLSTMsSequentialFCLsParallelFCLs(cp_ModelMethods):


    def __init__(
            self, n_features_inputs_lstm: int, n_features_outs_lstm: int,
            n_features_non_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            n_features_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            bias_lstm: typing.Union[bool, int] = True,
            biases_non_parallel_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            biases_parallel_fc_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            n_layers_lstm: int = 1, dropout_lstm: typing.Union[int, float] = 0, bidirectional_lstm: bool = False,
            batch_first: bool = True, return_hc: bool = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        superclass = SequentialLSTMsSequentialFCLsParallelFCLs
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

        # todo: homogeneous.SequentialLSTMs
        self.lstm = LSTMNNs()
        # self.lstm = LSTMNNs(n_features_inputs=n_features_inputs_lstm, n_features_outs=n_features_outs_lstm,
        #     n_layers=n_layers_lstm, bias=bias_lstm,
        #     dropout=dropout_lstm, bidirectional=bidirectional_lstm,
        #     batch_first=batch_first, return_hc=return_hc,
        #     device=self.device)

        self.non_parallel_fc_layers = FCNN(
            n_features_layers=n_features_non_parallel_fc_layers,
            biases_layers=biases_non_parallel_layers,
            device=self.device)

        if self.lstm.n_features_all_outs != self.non_parallel_fc_layers.n_features_layers[0]:
            raise ValueError('n_features_outs_lstm, n_features_non_parallel_fc_layers[0]')

        self.parallel_fc_layers = IndFCNNs(
            n_features_layers=n_features_parallel_fc_layers,
            biases_layers=biases_parallel_fc_layers, device=self.device)

        if self.non_parallel_fc_layers.n_features_layers[-1] != self.parallel_fc_layers.n_features_first_layers_together:
            raise ValueError('n_features_non_parallel_fc_layers[-1], n_features_parallel_fc_layers[0]')

        self.M = self.parallel_fc_layers.M

        self.return_hc = self.lstm.return_hc

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x: torch.Tensor, hc: typing.Union[tuple, list, None] = None):
        if self.return_hc:
            x, hc = self.lstm(x, hc)
            x = self.non_parallel_fc_layers(x)
            x = self.parallel_fc_layers(x)
            return x, hc
        else:
            x = self.lstm(x, hc)
            x = self.non_parallel_fc_layers(x)
            x = self.parallel_fc_layers(x)
            return x
