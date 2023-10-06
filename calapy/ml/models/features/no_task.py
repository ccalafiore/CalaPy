

from ... import torch
from ..model_tools import ModelMethods as cp_ModelMethods
import numpy as np
import typing

__all__ = [
    'SequentialFCLs', 'ParallelSequentialFCLs', 'LSTMSequentialParallelFCLs', 'SequentialParallelFCLs']


class SequentialFCLs(cp_ModelMethods):

    def __init__(
            self, n_features_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        superclass = SequentialFCLs
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
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x: torch.Tensor):
        if self.L > 1:
            return self.layers(x)
        else:
            return x


class ParallelSequentialFCLs(cp_ModelMethods):

    def __init__(
            self, n_features_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        superclass = ParallelSequentialFCLs
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
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

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


class LSTMSequentialParallelFCLs(cp_ModelMethods):

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

        superclass = LSTMSequentialParallelFCLs
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


class SequentialParallelFCLs(cp_ModelMethods):

    def __init__(
            self, n_features_non_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            n_features_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            biases_non_parallel_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            biases_parallel_fc_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            device: typing.Union[torch.device, str, None] = None) -> None:

        superclass = SequentialParallelFCLs
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
        self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def forward(self, x: torch.Tensor):

        x = self.non_parallel_fc_layers(x)
        x = self.parallel_fc_layers(x)
        return x
