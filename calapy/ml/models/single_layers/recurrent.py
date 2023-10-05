

import numpy as np
import torch
from .... import combinations as cp_combinations
from ..model_tools import ModelMethods as CPModelMethods

__all__ = ['RNN', 'LSTM', 'GRU']


class _Recurrent(CPModelMethods):

    def __init__(self, type_name, axis_time, h_sigma=0.1):

        """

        :type axis_time: int | None
        :type h_sigma: float | int | None

        """

        superclass = _Recurrent
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if CPModelMethods not in self.superclasses_initiated:
            CPModelMethods.__init__(self=self, device=None)
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
            elif type_name_f in self.accepted_type_names_with_hc:
                self.type_name = type_name_f
                self.is_with_hc = True
                self.init_h = self._init_hc
            else:
                raise ValueError('type_name')
        else:
            raise TypeError('type_name')

        if axis_time is None:
            self.axis_time = axis_time
            self.is_timed = False
            self.forward = self._forward_without_time_axis
        elif isinstance(axis_time, int):
            if axis_time < 0:
                raise ValueError('axis_time')
            else:
                self.axis_time = axis_time
                self.is_timed = True
                self.forward = self._forward_with_time_axis
        else:
            raise TypeError('axis_time')

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

    def _forward_with_time_axis(self, input, h=None):

        """

        :type input: torch.Tensor
        :type h: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims_plus_1:
            raise ValueError('input.ndim')
        else:
            if h is None:
                batch_shape = [input.shape[a] for a in range(0, input.ndim - 1, 1) if a != self.axis_time]
                h = self.init_h(batch_shape=batch_shape)

            T = time_size = input.shape[self.axis_time]

            outputs_shape = [input.shape[a] for a in range(0, input.ndim - 1, 1)] + [self.layer.hidden_size]
            outputs = torch.empty(size=outputs_shape, dtype=self.dtype, device=self.device, requires_grad=False)

            indexes_input_t = [slice(0, input.shape[a], 1) for a in range(0, input.ndim, 1)]  # type: list
            indexes_outputs_t = [slice(0, outputs.shape[a], 1) for a in range(0, outputs.ndim, 1)]  # type: list

            for t in range(0, T, 1):
                indexes_input_t[self.axis_time] = t
                tup_indexes_input_t = tuple(indexes_input_t)

                indexes_outputs_t[self.axis_time] = t
                tup_indexes_outputs_t = tuple(indexes_outputs_t)
                h = self._forward_without_time_axis(input=input[tup_indexes_input_t], h=h)

                if self.is_with_hc:
                    outputs[tup_indexes_outputs_t] = h[0]
                else:
                    outputs[tup_indexes_outputs_t] = h

            return outputs, h

    def _forward_without_time_axis(self, input, h=None):

        """

        :type input: torch.Tensor
        :type h: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims:
            raise ValueError('input.ndim')
        else:
            if h is None:
                batch_shape = [a for a in range(0, input.ndim - 1, 1)]
                h = self.init_h(batch_shape=batch_shape)

            if self.min_input_n_dims <= input.ndim <= self._torch_max_input_n_dims:
                return self.layer(input=input, hx=h)
            else:
                extra_batch_dims = input.ndim - self._torch_max_input_n_dims
                extra_batch_shape = input.shape[slice(0, extra_batch_dims, 1)]

                for indexes_i in cp_combinations.n_conditions_to_combinations_on_the_fly(
                        n_conditions=extra_batch_shape, dtype='i'):

                    tup_indexes_i = tuple(indexes_i.tolist())

                    if self.is_with_hc:
                        # h_i = h[0][tup_indexes_i], h[1][tup_indexes_i]
                        # h_i = self.layer(input=input[tup_indexes_i], hx=h_i)
                        # h[0][tup_indexes_i], h[1][tup_indexes_i] = h_i
                        h[0][tup_indexes_i], h[1][tup_indexes_i] = self.layer(
                            input=input[tup_indexes_i], hx=(h[0][tup_indexes_i], h[1][tup_indexes_i]))
                    else:
                        # h_i = h[tup_indexes_i]
                        # h_i = self.layer(input=input[tup_indexes_i], hx=h_i)
                        # h[tup_indexes_i] = h_i
                        h[tup_indexes_i] = self.layer(input=input[tup_indexes_i], hx=h[tup_indexes_i])

                return h

    def _init_h(self, batch_shape, generator=None):
        """

        :type batch_shape: int | list | tuple
        :type generator: torch.Generator | None
        :rtype: torch.Tensor
        """

        if isinstance(batch_shape, int):
            h_shape = [batch_shape, self.layer.hidden_size]
        elif isinstance(batch_shape, (list, tuple, torch.Size)):
            h_shape = batch_shape + [self.layer.hidden_size]
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

    def _init_hc(self, batch_shape, generators=None):
        """

        :type batch_shape: int | list | tuple
        :type generators:
            list[torch.Generator | None, torch.Generator | None] | tuple[torch.Generator | None, torch.Generator | None]
            | torch.Generator | None
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        if generators is None:
            generators = [None, None]
        elif isinstance(generators, torch.Generator):
            generators = [generators, generators]
        elif isinstance(generators, (list, tuple)):
            len_gens = len(generators)
            if len_gens == 0:
                generators = [None, None]
            elif len_gens == 1:
                generators = [generators[0], generators[0]]
            elif len_gens == 2:
                for g in range(0, 2, 1):
                    if generators[g] is not None and not isinstance(generators[g], torch.Generator):
                        raise TypeError(f'generators[{g:d}]')
            else:
                raise ValueError('len(generators)')
        else:
            raise TypeError('generators')

        h = (
            self._init_h(batch_shape=batch_shape, generator=generators[0]),
            self._init_h(batch_shape=batch_shape, generator=generators[1]))

        return h


class RNN(_Recurrent, torch.nn.RNNCell):

    def __init__(
            self, input_size, hidden_size, bias=True, nonlinearity='tanh', axis_time=0, h_sigma=0.1,
            device=None, dtype=None):

        superclass = RNN
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if _Recurrent not in self.superclasses_initiated:
            _Recurrent.__init__(self=self, type_name='rnn', axis_time=axis_time, h_sigma=h_sigma)
            if _Recurrent not in self.superclasses_initiated:
                self.superclasses_initiated.append(_Recurrent)

        # define attributes here
        self.layer = torch.nn.RNNCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity,
            device=device, dtype=dtype)

        self.get_device()
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class GRU(_Recurrent, torch.nn.GRUCell):

    def __init__(self, input_size, hidden_size, bias=True, axis_time=0, h_sigma=0.1, device=None, dtype=None):

        superclass = GRU
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if _Recurrent not in self.superclasses_initiated:
            _Recurrent.__init__(self=self, type_name='gru', axis_time=axis_time, h_sigma=h_sigma)
            if _Recurrent not in self.superclasses_initiated:
                self.superclasses_initiated.append(_Recurrent)

        # define attributes here
        self.layer = torch.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)

        self.get_device()
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class LSTM(_Recurrent, torch.nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True, axis_time=0, h_sigma=0.1, device=None, dtype=None):

        superclass = LSTM
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if _Recurrent not in self.superclasses_initiated:
            _Recurrent.__init__(self=self, type_name='lstm', axis_time=axis_time, h_sigma=h_sigma)
            if _Recurrent not in self.superclasses_initiated:
                self.superclasses_initiated.append(_Recurrent)

        # define attributes here
        self.layer = torch.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype)

        self.get_device()
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
