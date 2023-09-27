
import torch
from ... import combinations as cp_combinations


class _ConvNd:


    def __init__(
            self, nd, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        if isinstance(nd, int):
            if nd < 1:
                raise ValueError('nd')

            elif nd == 1:
                self.min_input_n_dims = 2
                self.max_input_n_dims = 3

            elif nd == 2:
                self.min_input_n_dims = 3
                self.max_input_n_dims = 4

            elif nd == 3:
                self.min_input_n_dims = 4
                self.max_input_n_dims = 5
            else:
                raise ValueError('nd')

        else:
            raise TypeError('nd')

        self.nd = nd


    def forward(self, input):

        """

        :type input: torch.Tensor

        :rtype: torch.Tensor

        """

        if input.ndim < self.min_input_n_dims:
            raise ValueError('input.ndim')
        elif self.min_input_n_dims <= input.ndim <= self.max_input_n_dims:
            return self._conv_forward(input, self.weight, self.bias)
        else:
            extra_batch_dims = input.ndim - self.max_input_n_dims
            extra_batch_shape = input.shape[slice(0, extra_batch_dims, 1)]
            is_output_initiated = False
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


class Conv1d(_ConvNd, torch.nn.Conv1d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        torch.nn.Conv1d.__init__(
            self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

        _ConvNd.__init__(
            self=self, nd=1, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device,
            dtype=dtype)


class Conv2d(_ConvNd, torch.nn.Conv2d):


    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        torch.nn.Conv2d.__init__(
            self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

        _ConvNd.__init__(
            self=self, nd=2, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device,
            dtype=dtype)


class Conv3d(_ConvNd, torch.nn.Conv3d):


    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        torch.nn.Conv3d.__init__(
            self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

        _ConvNd.__init__(
            self=self, nd=3, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device,
            dtype=dtype)