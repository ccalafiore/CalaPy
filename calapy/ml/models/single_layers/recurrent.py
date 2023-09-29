
import torch
from .... import combinations as cp_combinations


__all__ = ['RNN', 'LSTM', 'GRU']


class _Recurrent:


    def __init__(self):

        self.min_input_n_dims = 1
        self._torch_max_input_n_dims = 2

        self.nd = nd

    def forward(self, input):

        """

        :type input: torch.Tensor

        :rtype: torch.Tensor

        """

        if input.ndim < self.min_input_n_dims:
            raise ValueError('input.ndim')
        elif self.min_input_n_dims <= input.ndim <= self._torch_max_input_n_dims:
            return self._conv_forward(input, self.weight, self.bias)
        else:
            extra_batch_dims = input.ndim - self._torch_max_input_n_dims
            extra_batch_shape = input.shape[slice(0, extra_batch_dims, 1)]
            is_output_initiated = False
            output = None

            for indexes_i in cp_combinations.n_conditions_to_combinations_on_the_fly(
                    n_conditions=extra_batch_shape, dtype='i'):

                tup_indexes_i = tuple(indexes_i.tolist())

                if is_output_initiated:
                    output[tup_indexes_i] = self._conv_forward(input[tup_indexes_i], self.weight, self.bias)
                else:
                    output_i = self._conv_forward(input[tup_indexes_i], self.weight, self.bias)
                    output_shape = extra_batch_shape + output_i.shape
                    output = torch.empty(
                        size=output_shape, dtype=output_i.dtype, device=output_i.device, requires_grad=False)
                    is_output_initiated = True

                    output[tup_indexes_i] = output_i
                    del output_i

            return output

class RNN(_Recurrent, torch.nn.RNNCell):


    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        torch.nn.RNN.__init__(
            self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

        _Recurrent.__init__(self=self)
