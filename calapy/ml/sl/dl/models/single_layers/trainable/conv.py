

import torch
from ..... import combinations as cp_combinations
from ...model_tools import ModelMethods as CPModelMethods


__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class _ConvNd(CPModelMethods):

    def __init__(self, nd):

        superclass = _ConvNd
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

        if isinstance(nd, int):
            if nd < 1:
                raise ValueError('nd')

            elif nd == 1:
                self.min_input_n_dims = 2
                self._torch_max_input_n_dims = 3

            elif nd == 2:
                self.min_input_n_dims = 3
                self._torch_max_input_n_dims = 4

            elif nd == 3:
                self.min_input_n_dims = 4
                self._torch_max_input_n_dims = 5
            else:
                raise ValueError('nd')

        else:
            raise TypeError('nd')

        self.nd = nd

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

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


class Conv1d(_ConvNd, torch.nn.Conv1d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        superclass = Conv1d
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if torch.nn.Conv1d not in self.superclasses_initiated:
            torch.nn.Conv1d.__init__(
                self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                padding_mode=padding_mode, device=device, dtype=dtype)
            if torch.nn.Conv1d not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Conv1d)
            if torch.nn.Module not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Module)

        if _ConvNd not in self.superclasses_initiated:
            _ConvNd.__init__(self=self, nd=1)
            if _ConvNd not in self.superclasses_initiated:
                self.superclasses_initiated.append(_ConvNd)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class Conv2d(_ConvNd, torch.nn.Conv2d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        superclass = Conv2d
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if torch.nn.Conv2d not in self.superclasses_initiated:
            torch.nn.Conv2d.__init__(
                self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                padding_mode=padding_mode, device=device, dtype=dtype)
            if torch.nn.Conv2d not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Conv2d)
            if torch.nn.Module not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Module)

        if _ConvNd not in self.superclasses_initiated:
            _ConvNd.__init__(self=self, nd=2)
            if _ConvNd not in self.superclasses_initiated:
                self.superclasses_initiated.append(_ConvNd)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class Conv3d(_ConvNd, torch.nn.Conv3d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):

        superclass = Conv3d
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if torch.nn.Conv3d not in self.superclasses_initiated:
            torch.nn.Conv3d.__init__(
                self=self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                padding_mode=padding_mode, device=device, dtype=dtype)
            if torch.nn.Conv3d not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Conv3d)
            if torch.nn.Module not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Module)

        if _ConvNd not in self.superclasses_initiated:
            _ConvNd.__init__(self=self, nd=3)
            if _ConvNd not in self.superclasses_initiated:
                self.superclasses_initiated.append(_ConvNd)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
