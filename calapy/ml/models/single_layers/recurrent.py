

import numpy as np
import torch
from .... import combinations as cp_combinations
from ..model_tools import ModelMethods as CPModelMethods

__all__ = ['RNN', 'LSTM', 'GRU']


class _Recurrent(CPModelMethods):

    def __init__(self, axis_time, hx_sigma=0.1):

        """

        :type axis_time: int | None
        :type hx_sigma: float | int | None

        """

        superclass = _Recurrent
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if CPModelMethods not in self.superclasses_initiated:
            CPModelMethods.__init__(self=self, device=None)
            if CPModelMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(CPModelMethods)

        self.min_input_n_dims = 1
        self._torch_max_input_n_dims = 2

        self.min_input_n_dims_plus_1 = self.min_input_n_dims + 1
        self._torch_max_input_n_dims_plus_1 = self._torch_max_input_n_dims + 1

        if axis_time is None:
            self.axis_time = axis_time
            self.is_timed = False
        elif isinstance(axis_time, int):
            self.axis_time = axis_time
            self.is_timed = True
        else:
            raise TypeError('axis_time')

        if hx_sigma is None:
            self.hx_sigma = 0.0
        elif isinstance(hx_sigma, (int, float)):
            self.hx_sigma = hx_sigma
        else:
            raise TypeError('hx_sigma')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def _forward_with_time_axis(self, input, hx=None):

        """

        :type input: torch.Tensor
        :type hx: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims_plus_1:
            raise ValueError('input.ndim')
        else:
            if hx is None:
                batch_shape = [input.shape[a] for a in range(0, input.ndim - 1, 1) if a != self.axis_time]
                hx = self.init_hx(batch_shape=batch_shape)

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

                hx = self._forward_without_time_axis(input=input[tup_indexes_input_t], hx=hx)

                outputs[tup_indexes_outputs_t] = hx

            return outputs


    def _forward_without_time_axis(self, input, hx=None):

        """

        :type input: torch.Tensor
        :type hx: torch.Tensor
        :rtype: torch.Tensor
        """

        if input.ndim < self.min_input_n_dims:
            raise ValueError('input.ndim')
        else:
            if hx is None:
                batch_shape = [a for a in range(0, input.ndim - 1, 1)]
                hx = self.init_hx(batch_shape=batch_shape)
            if self.min_input_n_dims <= input.ndim <= self._torch_max_input_n_dims:
                return self.layer(input=input, hx=hx)
            else:
                extra_batch_dims = input.ndim - self._torch_max_input_n_dims
                extra_batch_shape = input.shape[slice(0, extra_batch_dims, 1)]

                for indexes_i in cp_combinations.n_conditions_to_combinations_on_the_fly(
                        n_conditions=extra_batch_shape, dtype='i'):
                    tup_indexes_i = tuple(indexes_i.tolist())
                    hx[tup_indexes_i] = self.layer(input=input[tup_indexes_i], hx=hx[tup_indexes_i])

                return hx

    def init_hx(self, batch_shape, generator=None):
        """

        :type batch_shape: int | list | tuple
        :type generator: torch.Generator | None
        :rtype: torch.Tensor
        """

        if isinstance(batch_shape, int):
            hx_shape = [batch_shape, self.layer.hidden_size]
        elif isinstance(batch_shape, (list, tuple)):
            hx_shape = batch_shape + [self.layer.hidden_size]
        elif isinstance(batch_shape, (torch.Tensor, np.ndarray)):
            hx_shape = batch_shape.tolist() + [self.layer.hidden_size]
        else:
            raise TypeError('batch_shape')

        if self.training:

            if self.hx_sigma == 0:
                hx = torch.zeros(size=hx_shape, dtype=self.dtype, device=self.device, requires_grad=False)
            else:
                hx = torch.randn(
                    size=hx_shape, generator=generator, dtype=self.dtype, device=self.device, requires_grad=False)
                if self.hx_sigma != 1.0:
                    hx *= self.hx_sigma
        else:
            hx = torch.zeros(size=hx_shape, dtype=self.dtype, device=self.device, requires_grad=False)

        return hx


class RNN(_Recurrent, torch.nn.RNNCell):

    def __init__(
            self, input_size, hidden_size, bias=True, nonlinearity='tanh', axis_time=0, hx_sigma=0.1,
            device=None, dtype=None):

        superclass = RNN
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if _Recurrent not in self.superclasses_initiated:
            _Recurrent.__init__(self=self, axis_time=axis_time, hx_sigma=hx_sigma)
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
