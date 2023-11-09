

import numpy as np
import torch
from ....... import combinations as cp_combinations
from ...model_tools import ModelMethods as CPModelMethods

__all__ = ['RNN', 'LSTM', 'GRU']


class _RNN(CPModelMethods):

    def __init__(self, type_name, axis_time, h_sigma=0.1):

        """

        :type axis_time: int | None
        :type h_sigma: float | int | None

        """

        superclass = _RNN
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

        self.accepted_type_names_with_h = tuple(['rnn', 'gru'])
        self.accepted_type_names_with_hc = tuple(['lstm'])
        self.accepted_type_names = self.accepted_type_names_with_h + self.accepted_type_names_with_hc
        if isinstance(type_name, str):
            type_name_f = type_name.lower()
            if type_name_f in self.accepted_type_names_with_h:
                self.type_name = type_name_f
                self.is_with_hc = False
                self.init_h = self._init_h
                # del self._init_hc
                self.concatenate_hs = self._concatenate_hs
                self.unbatch_h = self._unbatch_h
                self.get_batch_shape_from_h = self._get_batch_shape_from_h
            elif type_name_f in self.accepted_type_names_with_hc:
                self.type_name = type_name_f
                self.is_with_hc = True
                self.init_h = self._init_hc
                self.concatenate_hs = self._concatenate_hcs
                self.unbatch_h = self._unbatch_hc
                self.get_batch_shape_from_h = self._get_batch_shape_from_hc
            else:
                raise ValueError('type_name')
        else:
            raise TypeError('type_name')

        self.axis_time, self.is_timed, self.forward = self.set_axis_time(axis_time=axis_time)

        if h_sigma is None:
            self.h_sigma = 0.0
        elif isinstance(h_sigma, (int, float)):
            self.h_sigma = h_sigma
        else:
            raise TypeError('h_sigma')

        self.min_input_n_dims = 1
        self._torch_max_input_n_dims = 2

        self.min_input_n_dims_plus_1 = self.min_input_n_dims + 1
        self._torch_max_input_n_dims_plus_1 = self._torch_max_input_n_dims + 1

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def _forward_with_axis_time(self, input, h=None):

        """

        :type input: torch.Tensor
        :type h: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims_plus_1:
            raise ValueError('input.ndim')
        else:
            T = time_size = input.shape[self.axis_time]

            axis_features_input = input.ndim - 1
            if axis_features_input == self.axis_time:
                axis_features_input -= 1

            if h is None:
                batch_shape = self.get_batch_shape(input_shape=input.shape)
                h = self.init_h(batch_shape=batch_shape)

            # outputs_shape = [input.shape[a] for a in range(0, input.ndim, 1)]
            outputs_shape = list(input.shape)
            outputs_shape[axis_features_input] = self.layer.hidden_size
            outputs = torch.empty(size=outputs_shape, dtype=self.dtype, device=self.device, requires_grad=False)

            indexes_input_t = [slice(0, input.shape[a], 1) for a in range(0, input.ndim, 1)]  # type: list
            indexes_outputs_t = [slice(0, outputs.shape[a], 1) for a in range(0, outputs.ndim, 1)]  # type: list

            for t in range(0, T, 1):
                indexes_input_t[self.axis_time] = t
                tup_indexes_input_t = tuple(indexes_input_t)

                indexes_outputs_t[self.axis_time] = t
                tup_indexes_outputs_t = tuple(indexes_outputs_t)
                h = self._forward_without_axis_time(input=input[tup_indexes_input_t], h=h)

                if self.is_with_hc:
                    outputs[tup_indexes_outputs_t] = h[0]
                else:
                    outputs[tup_indexes_outputs_t] = h

            return outputs, h

    def _forward_without_axis_time(self, input, h=None):

        """

        :type input: torch.Tensor
        :type h: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims:
            raise ValueError('input.ndim')
        else:
            if h is None:
                batch_shape = self.get_batch_shape(input_shape=input.shape)
                h = self.init_h(batch_shape=batch_shape)

            if self.min_input_n_dims <= input.ndim <= self._torch_max_input_n_dims:
                return self.layer(input=input, hx=h)
            else:
                n_extra_batch_dims = input.ndim - self._torch_max_input_n_dims
                extra_batch_shape = input.shape[slice(0, n_extra_batch_dims, 1)]

                indexes_input_i = np.asarray([slice(0, input.shape[a], 1) for a in range(0, input.ndim, 1)], dtype='O')

                if self.is_with_hc:
                    h_shape = h[0].shape
                    h_ndim = h[0].ndim
                else:
                    h_shape = h.shape
                    h_ndim = h.ndim
                indexes_h_i = np.asarray([slice(0, h_shape[a], 1) for a in range(0, h_ndim, 1)], dtype='O')

                for indexes_i in cp_combinations.n_conditions_to_combinations_on_the_fly(
                        n_conditions=extra_batch_shape, dtype='i'):

                    indexes_input_i[slice(0, n_extra_batch_dims, 1)] = indexes_i
                    tup_indexes_input_i = tuple(indexes_input_i.tolist())

                    indexes_h_i[slice(0, n_extra_batch_dims, 1)] = indexes_i
                    tup_indexes_h_i = tuple(indexes_h_i.tolist())

                    if self.is_with_hc:
                        # h_i = h[0][tup_indexes_h_i], h[1][tup_indexes_h_i]
                        # h_i = self.layer(input=input[tup_indexes_input_i], hx=h_i)
                        # h[0][tup_indexes_h_i], h[1][tup_indexes_h_i] = h_i
                        h[0][tup_indexes_h_i], h[1][tup_indexes_h_i] = self.layer(
                            input=input[tup_indexes_input_i], hx=(h[0][tup_indexes_h_i], h[1][tup_indexes_h_i]))
                    else:
                        # h_i = h[tup_indexes_h_i]
                        # h_i = self.layer(input=input[tup_indexes_input_i], hx=h_i)
                        # h[tup_indexes_h_i] = h_i
                        h[tup_indexes_h_i] = self.layer(input=input[tup_indexes_input_i], hx=h[tup_indexes_h_i])

                return h

    def _init_h(self, batch_shape, generator=None):
        """

        :type batch_shape: int | list | tuple | torch.Size | torch.Tensor, np.ndarray
        :type generator: torch.Generator | None
        :rtype: torch.Tensor
        """

        if isinstance(batch_shape, int):
            h_shape = [batch_shape, self.layer.hidden_size]
        elif isinstance(batch_shape, list):
            h_shape = batch_shape + [self.layer.hidden_size]
        elif isinstance(batch_shape, (tuple, torch.Size)):
            h_shape = list(batch_shape) + [self.layer.hidden_size]
        elif isinstance(batch_shape, (torch.Tensor, np.ndarray)):
            h_shape = batch_shape.tolist() + [self.layer.hidden_size]
        else:
            raise TypeError('batch_shape')

        if self.training:

            if self.h_sigma == 0:
                h = torch.zeros(size=h_shape, dtype=self.dtype, device=self.device, requires_grad=False)
            else:
                h = torch.randn(
                    size=h_shape, generator=generator, dtype=self.dtype, device=self.device, requires_grad=False)
                if self.h_sigma != 1.0:
                    h *= self.h_sigma
        else:
            h = torch.zeros(size=h_shape, dtype=self.dtype, device=self.device, requires_grad=False)
        return h

    def _init_hc(self, batch_shape, generator=None):
        """

        :type batch_shape: int | list | tuple | torch.Size | torch.Tensor, np.ndarray
        :type generator:
            list[torch.Generator | None, torch.Generator | None] | tuple[torch.Generator | None, torch.Generator | None]
            | torch.Generator | None
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        if generator is None:
            generator = [None, None]
        elif isinstance(generator, torch.Generator):
            generator = [generator, generator]
        elif isinstance(generator, (list, tuple)):
            len_gens = len(generator)
            if len_gens == 0:
                generator = [None, None]
            elif len_gens == 1:
                generator = [generator[0], generator[0]]
            elif len_gens == 2:
                for g in range(0, 2, 1):
                    if generator[g] is not None and not isinstance(generator[g], torch.Generator):
                        raise TypeError(f'generator[{g:d}]')
            else:
                raise ValueError('len(generator)')
        else:
            raise TypeError('generator')

        h = (
            self._init_h(batch_shape=batch_shape, generator=generator[0]),
            self._init_h(batch_shape=batch_shape, generator=generator[1]))

        return h

    def get_batch_shape_from_input_shape(self, input_shape):

        """

        :param input_shape: The input shape.
        :type input_shape: int | list | tuple | torch.Tensor | np.ndarray
        :return: The batch shape given the input shape "input_shape" and the recurrent model.
        :rtype: list[int]
        """

        if isinstance(input_shape, int):
            input_shape_f = [input_shape]
        elif isinstance(input_shape, list):
            input_shape_f = input_shape
        elif isinstance(input_shape, (tuple, torch.Size)):
            input_shape_f = list(input_shape)
        elif isinstance(input_shape, (torch.Tensor, np.ndarray)):
            input_shape_f = input_shape.tolist()
        else:
            raise TypeError('input_shape')

        n_dims_input = len(input_shape_f)
        axis_features_input = n_dims_input - 1

        if self.is_timed:
            if axis_features_input == self.axis_time:
                axis_features_input -= 1
            batch_shape = [
                input_shape_f[a] for a in range(0, n_dims_input, 1) if a not in [self.axis_time, axis_features_input]]
        else:
            batch_shape = [input_shape_f[a] for a in range(0, n_dims_input, 1) if a != axis_features_input]

        return batch_shape

    def set_axis_time(self, axis_time):

        """

        :type axis_time: int | None
        """

        if axis_time is None:
            self.axis_time = axis_time
            self.is_timed = False
            self.forward = self._forward_without_axis_time
        elif isinstance(axis_time, int):
            if axis_time < 0:
                raise ValueError('axis_time')
            else:
                self.axis_time = axis_time
                self.is_timed = True
                self.forward = self._forward_with_axis_time
        else:
            raise TypeError('axis_time')

        return self.axis_time, self.is_timed, self.forward

    def _concatenate_hs(self, hs, axis=0):

        return torch.cat(hs, dim=axis)

    def _concatenate_hcs(self, hs, axis=0):

        return [torch.cat(hs[0], dim=axis), torch.cat(hs[1], dim=axis)]

    def _unbatch_h(self, h, axes=0, keepdims=True):

        """


        :type h: torch.Tensor
        :type axes: list | tuple | int
        :type keepdims: bool

        """

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, (list, tuple)):
            pass
        else:
            raise TypeError('axes')

        batch_shape = self._get_batch_shape_from_h(h=h, axes=axes)

        indexes = np.asarray([slice(0, h.shape[a], 1) for a in range(0, h.ndim, 1)], dtype='O')

        hs = np.empty(shape=batch_shape, dtype='O')
        for comps_i in cp_combinations.n_conditions_to_combinations_on_the_fly(n_conditions=batch_shape, dtype='i'):

            if keepdims:
                ind_i = [slice(comp_i, comp_i + 1, 1) for comp_i in comps_i.tolist()]
            else:
                ind_i = comps_i.tolist()

            indexes[axes] = ind_i

            hs[tuple(comps_i)] = h[tuple(indexes)]

        return hs.tolist()

    def _unbatch_hc(self, h, axes=0, keepdims=True):

        """


        :type h: list[torch.Tensor, torch.Tensor]
        :type axes: list | tuple | int
        :type keepdims: bool

        """

        Z = 2

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, (list, tuple)):
            pass
        else:
            raise TypeError('axes')

        batch_shape = self._get_batch_shape_from_hc(h=h, axes=axes)

        hs = np.empty(shape=batch_shape + [Z], dtype='O')
        indexes_hs = [slice(0, hs.shape[a], 1) for a in range(0, hs.ndim, 1)]  # type: list

        for z in range(0, Z, 1):
            indexes_hs[hs.ndim - 1] = z
            hs[tuple(indexes_hs)] = self._unbatch_h(h[z], axes=axes, keepdims=keepdims)

        return hs.tolist()

    def _get_batch_shape_from_h(self, h, axes):

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, (list, tuple)):
            pass
        else:
            raise TypeError('axes')

        batch_shape = [h.shape[a] for a in axes]
        return batch_shape

    def _get_batch_shape_from_hc(self, h, axes):

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, (list, tuple)):
            pass
        else:
            raise TypeError('axes')

        batch_shape = [h[0].shape[a] for a in axes]
        return batch_shape




class RNN(_RNN):

    def __init__(
            self, input_size, hidden_size, bias=True, axis_time=None, h_sigma=0.1, nonlinearity='tanh',
            device=None, dtype=None):

        superclass = RNN
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if _RNN not in self.superclasses_initiated:
            _RNN.__init__(self=self, type_name='rnn', axis_time=axis_time, h_sigma=h_sigma)
            if _RNN not in self.superclasses_initiated:
                self.superclasses_initiated.append(_RNN)

        # define attributes here
        self.layer = torch.nn.RNNCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity,
            device=device, dtype=dtype)

        if superclass == type(self):
            self.get_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class GRU(_RNN):

    def __init__(self, input_size, hidden_size, bias=True, axis_time=None, h_sigma=0.1, device=None, dtype=None):

        superclass = GRU
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if _RNN not in self.superclasses_initiated:
            _RNN.__init__(self=self, type_name='gru', axis_time=axis_time, h_sigma=h_sigma)
            if _RNN not in self.superclasses_initiated:
                self.superclasses_initiated.append(_RNN)

        # define attributes here
        self.layer = torch.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)

        if superclass == type(self):
            self.get_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class LSTM(_RNN):

    def __init__(self, input_size, hidden_size, bias=True, axis_time=None, h_sigma=0.1, device=None, dtype=None):

        superclass = LSTM
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if _RNN not in self.superclasses_initiated:
            _RNN.__init__(self=self, type_name='lstm', axis_time=axis_time, h_sigma=h_sigma)
            if _RNN not in self.superclasses_initiated:
                self.superclasses_initiated.append(_RNN)

        # define attributes here
        self.layer = torch.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)

        if superclass == type(self):
            self.get_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
