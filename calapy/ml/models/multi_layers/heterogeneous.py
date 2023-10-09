

import numpy as np
import torch
from ..model_tools import ModelMethods as CPModelMethods
from .. import single_layers as cp_single_layers
from ... import tensor as cp_tensor

__all__ = ['SequentialMultiHeteroLayers']


class SequentialMultiHeteroLayers(CPModelMethods):

    """The class of basic neural networks (NNs)

    """

    def __init__(self, params_of_layers, device=None, dtype=None):

        """

        :param params_of_layers:
        :type params_of_layers: list[dict] | tuple[dict] | np.ndarray[dict]
        :type device: torch.device | str | int | None
        :type dtype: torch.dtype | str| None

        """

        superclass = SequentialMultiHeteroLayers
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
            ['noise', 'dropout', 'sigmoid', 'tanh', 'relu', 'flatten'])
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

        self.device = device
        self.dtype = dtype

        # define self.layers
        self.layers = torch.nn.ModuleList()
        self.hidden_state_sizes = []
        self.recurrent_layer_indexes = []
        self.recurrent_layer_types = []

        for l in range(0, self.L, 1):
            if self.params_of_layers[l]['type_name'] == 'fc':
                layer_l = torch.nn.Linear(**self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
            elif self.params_of_layers[l]['type_name'] == 'rnn':
                layer_l = cp_single_layers.RNN(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'lstm':
                layer_l = cp_single_layers.LSTM(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'gru':
                layer_l = cp_single_layers.GRU(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
                self.hidden_state_sizes.append(self.params_of_layers[l]['params']['hidden_size'])
                self.recurrent_layer_indexes.append(l)
                self.recurrent_layer_types.append(self.params_of_layers[l]['type_name'])
            elif self.params_of_layers[l]['type_name'] == 'conv1d':
                layer_l = cp_single_layers.Conv1d(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
            elif self.params_of_layers[l]['type_name'] == 'conv2d':
                layer_l = cp_single_layers.Conv2d(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
            elif self.params_of_layers[l]['type_name'] == 'conv3d':
                layer_l = cp_single_layers.Conv3d(
                    **self.params_of_layers[l]['params'], device=self.device, dtype=self.dtype)
            elif self.params_of_layers[l]['type_name'] == 'noise':
                layer_l = cp_single_layers.NoiseLayer(**self.params_of_layers[l]['params'])
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

        if not isinstance(self.device, torch.device):
            self.get_device()
        if not isinstance(self.dtype, torch.dtype):
            self.get_dtype()

        self.Z = self.n_recurrent_layers = len(self.hidden_state_sizes)
        self.is_with_any_recurrent_layers = self.Z > 0

        if self.is_with_any_recurrent_layers:
            self.forward = self._forward_with_recurrent_layers
        else:
            self.forward = self._forward_without_recurrent_layers

    def _forward_without_recurrent_layers(self, x):

        """

        :type x: torch.Tensor | np.ndarray
        """

        for l in range(0, self.L, 1):
            x = self.layers[l](x)

        return x

    def _forward_with_recurrent_layers(self, x, h=None):

        """

        :type x: torch.Tensor | np.ndarray
        :type h: list[list[np.ndarray | torch.Tensor | None] | tuple[np.ndarray | torch.Tensor | None] | np.ndarray | torch.Tensor | None] | None
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
