

from ... import torch
from ..model_tools import ModelMethods as cc_ModelMethods
import numpy as np
import typing

__all__ = [
    'SequentialFCLs', 'ParallelSequentialFCLs', 'LSTM', 'LSTMSequentialParallelFCLs', 'SequentialParallelFCLs']


class SequentialFCLs(cc_ModelMethods):

    def __init__(
            self, n_features_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        name_superclass = SequentialFCLs.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if cc_ModelMethods.__name__ not in self.superclasses_initiated:
            cc_ModelMethods.__init__(self=self, device=device)

        self.n_outputs = self.O = self.n_models = self.M = 1

        if isinstance(n_features_layers, int):
            self.n_features_layers = [n_features_layers]  # type: list
        elif isinstance(n_features_layers, list):
            self.n_features_layers = n_features_layers
        elif isinstance(n_features_layers, tuple):
            self.n_features_layers = list(n_features_layers)
        elif isinstance(n_features_layers, (np.ndarray, torch.Tensor)):
            self.n_features_layers = n_features_layers.tolist()
        else:
            raise TypeError('n_features_layers')

        self.n_layers = self.L = len(self.n_features_layers)

        self.n_features_first_layer = self.n_features_layers[0]

        if isinstance(biases_layers, bool):
            self.biases_layers = [biases_layers for l in range(0, self.L - 1, 1)]  # type: list
        elif isinstance(biases_layers, int):
            biases_layers_l = bool(biases_layers)
            self.biases_layers = [biases_layers_l for l in range(0, self.L - 1, 1)]  # type: list
        elif isinstance(biases_layers, (list, tuple, np.ndarray, torch.Tensor)):
            tmp_len_biases_layers = len(biases_layers)
            if tmp_len_biases_layers == self.L - 1:
                if isinstance(biases_layers, list):
                    self.biases_layers = biases_layers
                elif isinstance(biases_layers, tuple):
                    self.biases_layers = list(biases_layers)
                elif isinstance(biases_layers, (np.ndarray, torch.Tensor)):
                    self.biases_layers = biases_layers.tolist()
            elif tmp_len_biases_layers == 1:
                if isinstance(biases_layers, (list, tuple)):
                    self.biases_layers = [biases_layers[0] for l in range(0, self.L - 1, 1)]
                elif isinstance(biases_layers, (np.ndarray, torch.Tensor)):
                    self.biases_layers = [biases_layers[0].tolist() for l in range(0, self.L - 1, 1)]
        else:
            raise TypeError('biases_layers')

        self.layers = torch.nn.Sequential(*[torch.nn.Linear(
            self.n_features_layers[l - 1], self.n_features_layers[l],
            bias=self.biases_layers[l - 1], device=self.device) for l in range(1, self.L, 1)])

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

    def forward(self, x: torch.Tensor):
        if self.L > 1:
            return self.layers(x)
        else:
            return x


class ParallelSequentialFCLs(cc_ModelMethods):

    def __init__(
            self, n_features_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        name_superclass = ParallelSequentialFCLs.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if cc_ModelMethods.__name__ not in self.superclasses_initiated:
            cc_ModelMethods.__init__(self=self, device=device)

        if isinstance(n_features_layers, int):
            self.n_features_layers = [[n_features_layers]]  # type: list
            self.n_outputs = self.O = self.n_models = self.M = 1
        elif isinstance(n_features_layers, (list, tuple, np.ndarray, torch.Tensor)):
            self.n_outputs = self.O = self.n_models = self.M = len(n_features_layers)
            self.n_features_layers = [None for m in range(0, self.M, 1)]  # type: list
            for m in range(0, self.M, 1):
                if isinstance(n_features_layers[m], int):
                    self.n_features_layers[m] = [n_features_layers[m]]
                elif isinstance(n_features_layers[m], list):
                    self.n_features_layers[m] = n_features_layers[m]
                elif isinstance(n_features_layers[m], tuple):
                    self.n_features_layers[m] = list(n_features_layers[m])
                elif isinstance(n_features_layers[m], (np.ndarray, torch.Tensor)):
                    self.n_features_layers[m] = n_features_layers[m].tolist()
                else:
                    raise TypeError('n_features_layers[' + str(m) + ']')
        else:
            raise TypeError('n_features_layers')

        self.n_layers = self.L = np.asarray(
            [len(self.n_features_layers[m]) for m in range(0, self.M, 1)], dtype='i')

        self.n_features_first_layers = [self.n_features_layers[m][0] for m in range(0, self.M, 1)]
        self.n_features_first_layers_together = sum(self.n_features_first_layers)

        self.n_features_last_layers = [self.n_features_layers[m][-1] for m in range(0, self.M, 1)]
        self.n_features_last_layers_together = sum(self.n_features_last_layers)

        if isinstance(biases_layers, bool):
            self.biases_layers = [
                [biases_layers for l in range(0, self.L[m] - 1, 1)] for m in range(0, self.M, 1)]  # type: list
        elif isinstance(biases_layers, int):
            biases_layers_ml = bool(biases_layers)
            self.biases_layers = [
                [biases_layers_ml for l in range(0, self.L[m] - 1, 1)] for m in range(0, self.M, 1)]  # type: list
        elif isinstance(biases_layers, (list, tuple, np.ndarray, torch.Tensor)):
            tmp_M = len(biases_layers)
            if (tmp_M == self.M) or (tmp_M == 1):
                index_m = 0
            else:
                raise ValueError('biases_layers = ' + str(biases_layers))
            self.biases_layers = [None for m in range(0, self.M, 1)]  # type: list
            for m in range(0, self.M, 1):
                if tmp_M == self.M:
                    index_m = m
                if isinstance(biases_layers[index_m], bool):
                    self.biases_layers[m] = [biases_layers[index_m] for l in range(0, self.L[m] - 1, 1)]
                elif isinstance(biases_layers[index_m], int):
                    biases_layers_m = bool(biases_layers[index_m])
                    self.biases_layers[m] = [biases_layers_m for l in range(0, self.L[m] - 1, 1)]
                elif isinstance(biases_layers[m], (list, tuple, np.ndarray, torch.Tensor)):
                    tmp_len_biases_layers_m = len(biases_layers[m])
                    if tmp_len_biases_layers_m == self.L[m] - 1:
                        if isinstance(biases_layers[index_m], list):
                            self.biases_layers[m] = biases_layers[index_m]
                        elif isinstance(biases_layers[index_m], tuple):
                            self.biases_layers[m] = list(biases_layers[index_m])
                        elif isinstance(biases_layers[index_m], (np.ndarray, torch.Tensor)):
                            self.biases_layers[m] = biases_layers[index_m].tolist()
                    elif tmp_len_biases_layers_m == 1:
                        if isinstance(biases_layers[index_m], (list, tuple)):
                            self.biases_layers[m] = [
                                biases_layers[index_m][0] for l in range(0, self.L[m] - 1, 1)]
                        elif isinstance(biases_layers[index_m], (np.ndarray, torch.Tensor)):
                            self.biases_layers[m] = [
                                biases_layers[index_m][0].tolist() for l in range(0, self.L[m] - 1, 1)]
                    else:
                        raise ValueError('len(biases_layers[' + str(m) + '])')
                else:
                    raise TypeError('biases_layers[' + str(m) + ']')
        else:
            raise TypeError('biases_layers')

        self.layers = torch.nn.ModuleList([torch.nn.Sequential(
            *[torch.nn.Linear(
                self.n_features_layers[m][l - 1], self.n_features_layers[m][l],
                bias=self.biases_layers[m][l - 1], device=self.device)
                for l in range(1, self.L[m], 1)]) for m in range(0, self.M, 1)])

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

    def forward(self, x: typing.Union[torch.Tensor, list, tuple], axis_features: typing.Optional[int] = None):

        if isinstance(x, torch.Tensor):
            if axis_features is None:
                axis_features = x.ndim - 1
            x = x.split(self.n_features_first_layers, dim=axis_features)
        elif isinstance(x, (list, tuple)):
            pass
        else:
            raise TypeError('type(x) = {}'.format(type(x)))

        outs = [self.layers[m](x[m]) if self.L[m] > 1 else x[m] for m in range(0, self.M, 1)]

        return outs


class LSTM(cc_ModelMethods):
    # todo sequential LSTMs

    def __init__(
            self, n_features_inputs: int, n_features_outs: int, bias: typing.Union[bool, int] = True,
            n_layers: int = 1, dropout: typing.Union[int, float] = 0, bidirectional: bool = False,
            batch_first: bool = False, return_hc: bool = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        name_superclass = LSTM.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if cc_ModelMethods.__name__ not in self.superclasses_initiated:
            cc_ModelMethods.__init__(self=self, device=device)

        self.n_features_inputs = n_features_inputs
        self.n_features_outs = n_features_outs

        self.n_layers = n_layers

        if isinstance(bias, bool):
            self.bias = bias
        elif isinstance(bias, int):
            self.bias = bool(bias)
        else:
            raise TypeError('bias = ' + str(bias))

        self.batch_first = batch_first

        self.dropout = dropout

        self.bidirectional = bidirectional

        # lstm tutorials at:
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

        self.lstm = torch.nn.LSTM(
            self.n_features_inputs, self.n_features_outs,
            num_layers=self.n_layers, bias=self.bias,
            batch_first=self.batch_first, dropout=self.dropout,
            bidirectional=self.bidirectional, device=self.device)

        self.return_hc = return_hc

        if self.bidirectional:
            self.n_outs = self.num_directions = 2
        else:
            self.n_outs = self.num_directions = 1

        self.n_features_all_outs = self.n_features_outs * self.n_outs

        if self.batch_first:
            self.axis_batch_inputs = 0
            self.axis_time_inputs = 1
        else:
            self.axis_batch_inputs = 1
            self.axis_time_inputs = 0
        self.axis_features_inputs = 2

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

    def forward(self, x: torch.Tensor, hc: typing.Union[tuple, list, None] = None):

        if hc is None:
            batch_size = x.shape[self.axis_batch_inputs]
            hc = self.init_hc(batch_size)
        elif (hc[0] is None) or (hc[1] is None):
            batch_size = x.shape[self.axis_batch_inputs]
            h, c = hc
            if h is None:
                h = self.init_h(batch_size)
            if c is None:
                c = self.init_c(batch_size)
            hc = h, c

        x, hc = self.lstm(x, hc)

        if self.return_hc:
            return x, hc
        else:
            return x

    def init_h(self, batch_size):

        h = torch.zeros(
            [self.num_directions * self.n_layers, batch_size, self.n_features_outs],
            dtype=torch.float32, device=self.device, requires_grad=False)

        return h

    def init_c(self, batch_size):

        c = torch.zeros(
            [self.num_directions * self.n_layers, batch_size, self.n_features_outs],
            dtype=torch.float32, device=self.device, requires_grad=False)

        return c

    def init_hc(self, batch_size):

        hc = self.init_h(batch_size), self.init_c(batch_size)

        return hc


class LSTMSequentialParallelFCLs(cc_ModelMethods):

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

        name_superclass = LSTMSequentialParallelFCLs.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if cc_ModelMethods.__name__ not in self.superclasses_initiated:
            cc_ModelMethods.__init__(self=self, device=device)

        self.lstm = LSTM(
            n_features_inputs=n_features_inputs_lstm, n_features_outs=n_features_outs_lstm,
            n_layers=n_layers_lstm, bias=bias_lstm,
            dropout=dropout_lstm, bidirectional=bidirectional_lstm,
            batch_first=batch_first, return_hc=return_hc,
            device=self.device)

        self.non_parallel_fc_layers = SequentialFCLs(
            n_features_layers=n_features_non_parallel_fc_layers,
            biases_layers=biases_non_parallel_layers,
            device=self.device)

        if self.lstm.n_features_all_outs != self.non_parallel_fc_layers.n_features_layers[0]:
            raise ValueError('n_features_outs_lstm, n_features_non_parallel_fc_layers[0]')

        self.parallel_fc_layers = ParallelSequentialFCLs(
            n_features_layers=n_features_parallel_fc_layers,
            biases_layers=biases_parallel_fc_layers, device=self.device)

        if self.non_parallel_fc_layers.n_features_layers[-1] != self.parallel_fc_layers.n_features_first_layers_together:
            raise ValueError('n_features_non_parallel_fc_layers[-1], n_features_parallel_fc_layers[0]')

        self.M = self.parallel_fc_layers.M

        self.return_hc = self.lstm.return_hc

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

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


class SequentialParallelFCLs(cc_ModelMethods):

    def __init__(
            self, n_features_non_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            n_features_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_non_parallel_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            biases_parallel_fc_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        name_superclass = SequentialParallelFCLs.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if cc_ModelMethods.__name__ not in self.superclasses_initiated:
            cc_ModelMethods.__init__(self=self, device=device)

        self.non_parallel_fc_layers = SequentialFCLs(
            n_features_layers=n_features_non_parallel_fc_layers,
            biases_layers=biases_non_parallel_layers,
            device=self.device)

        self.parallel_fc_layers = ParallelSequentialFCLs(
            n_features_layers=n_features_parallel_fc_layers,
            biases_layers=biases_parallel_fc_layers, device=self.device)

        if self.non_parallel_fc_layers.n_features_layers[-1] != self.parallel_fc_layers.n_features_first_layers_together:
            raise ValueError('n_features_non_parallel_fc_layers[-1], n_features_parallel_fc_layers[0]')

        self.M = self.parallel_fc_layers.M

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

    def forward(self, x: torch.Tensor):

        x = self.non_parallel_fc_layers(x)
        x = self.parallel_fc_layers(x)
        return x
