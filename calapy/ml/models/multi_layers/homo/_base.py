

import numpy as np
import torch
from ...model_tools import ModelMethods as cp_ModelMethods
# from ..... import combinations as cp_combinations


__all__ = ['_NN', '_IndNNs']

# todo: add non-trainable layers


class _NN(cp_ModelMethods):

    """A Base Class for neural networks (NN) with homogeneous layers.


    """

    def __init__(self, n_features_layers, biases_layers=True):

        """

        :param n_features_layers: A sequence of ints whose first element is the input size of the NN and all other
            elements from the second to the last are the output sizes of all layers. The NN will have L layers, where
            L=len(n_features_layers)-1. The n_features_layers must have 2 elements or more such that the NN can have 1
            layer or more.
        :type n_features_layers: list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]

        :param biases_layers: A bool or a sequence of bools with L elements where L is the number of trainable layers.
            If it is a bool and is True (default), all layers WILL HAVE trainable biases. If it is a bool and is False,
            all layers WILL NOT HAVE trainable biases. If it is a 1d vector of L bools and biases_layers[l] is
            True, the l_th layer WILL HAVE trainable biases. If it is a 1d vector of L bools and biases_layers[l] is
            False, then the l_th layer WILL NOT HAVE trainable biases.
        :type biases_layers: bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]
        """

        superclass = _NN
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

        if isinstance(n_features_layers, list):
            self.n_features_layers = n_features_layers
        elif isinstance(n_features_layers, tuple):
            self.n_features_layers = list(n_features_layers)
        elif isinstance(n_features_layers, (np.ndarray, torch.Tensor)):
            self.n_features_layers = n_features_layers.tolist()
        else:
            raise TypeError('n_features_layers')

        self.n_layers = self.L = len(self.n_features_layers) - 1

        if self.L < 1:
            raise ValueError('n_features_layers needs to be a 1d vector of 2 elements of more')

        self.n_input_features_nn = self.n_features_layers[0]
        self.n_output_features_nn = self.n_features_layers[-1]

        self.n_input_features_layers = self.n_features_layers[slice(0, self.L, 1)]
        self.n_output_features_layers = self.n_features_layers[slice(1, self.L + 1, 1)]

        if isinstance(biases_layers, bool):
            self.biases_layers = [biases_layers for l in range(0, self.L, 1)]  # type: list
        elif isinstance(biases_layers, (list, tuple, np.ndarray, torch.Tensor)):
            tmp_len_biases_layers = len(biases_layers)
            if tmp_len_biases_layers == self.L:
                if isinstance(biases_layers, list):
                    self.biases_layers = biases_layers
                elif isinstance(biases_layers, tuple):
                    self.biases_layers = list(biases_layers)
                elif isinstance(biases_layers, (np.ndarray, torch.Tensor)):
                    self.biases_layers = biases_layers.tolist()
            elif tmp_len_biases_layers == 1:
                if isinstance(biases_layers, (list, tuple)):
                    biases_layers_f = biases_layers[0]
                elif isinstance(biases_layers, (np.ndarray, torch.Tensor)):
                    biases_layers_f = biases_layers[0].tolist()
                else:
                    raise TypeError()
                self.biases_layers = [biases_layers_f for l in range(0, self.L, 1)]
            else:
                raise ValueError('biases_layers')
        else:
            raise TypeError('biases_layers')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)


class _IndNNs(cp_ModelMethods):

    """A Class of multiple independent neural networks (NN) with only fully-connected (FC) layers.

    """

    def __init__(self, n_features_layers, biases_layers=True, axis_features=None):

        """

        :param n_features_layers: A sequence of M sequences, one per neural network (NN), where M is the number of
            independent NNs. The M sequences can have different sizes, but each size should be 2 or more. The first
            element of the m_th sequence is the input size of the m_th NN, while the other elements from the second to
            the last of the m_th sequence are the output sizes (the actual layer sizes) of all layers of the m_th NN.
            The m_th NN will have L[m] layers, where L[m]=len(n_features_layers[m])-1.
        :type n_features_layers: list[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]] |
                                 tuple[list[int] | tuple[int] | np.ndarray[int] | torch.Tensor[int]]

        :param biases_layers: A bool, sequence of M bools, or a sequence of M sequences of L[m] bools where M is
            the number of independent models (or NNs) with private deeper layers and L[m] is the number of private
            layers of the m_th independent model. If it is a bool and is True (default), all layers WILL HAVE trainable
            biases. If it is a bool and is False, all layers WILL NOT HAVE trainable biases. If it is a sequence of M
            bools and biases_layers[m] is True, then all layers of the m_th NN WILL HAVE trainable biases. If it is a
            sequence of M bools and biases_layers[m] is False, then all layers of the m_th NN WILL NOT HAVE trainable
            biases. If it is a sequence of M sequences of L[m] bools and biases_layers[m][l] is True, then the l_th
            layer of the m_th model WILL HAVE trainable biases. If it is a sequence of M sequences of L[m] bools and
            biases_layers[m][l] is False, then the l_th layer of the m_th model WILL NOT HAVE trainable biases.
        :type biases_layers: bool | list[bool] | tuple[bool] | np.ndarray[bool] | torch.Tensor[bool]
        """

        superclass = _IndNNs
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

        if isinstance(n_features_layers, (list, tuple, np.ndarray, torch.Tensor)):
            self.n_outputs = self.O = self.n_models = self.M = len(n_features_layers)
            self.n_features_layers = [None for m in range(0, self.M, 1)]  # type: list
            for m in range(0, self.M, 1):
                if isinstance(n_features_layers[m], list):
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
            [len(self.n_features_layers[m]) - 1 for m in range(0, self.M, 1)], dtype='i')

        self.n_features_first_layers = [self.n_features_layers[m][0] for m in range(0, self.M, 1)]
        self.n_features_first_layers_together = sum(self.n_features_first_layers)

        self.n_features_last_layers = [self.n_features_layers[m][-1] for m in range(0, self.M, 1)]
        self.n_features_last_layers_together = sum(self.n_features_last_layers)

        self.n_input_features_layers = [self.n_features_layers[m][slice(0, self.L[m], 1)] for m in range(0, self.M, 1)]
        self.n_output_features_layers = [
            self.n_features_layers[m][slice(1, self.L[m] + 1, 1)] for m in range(0, self.M, 1)]

        if isinstance(biases_layers, bool):
            self.biases_layers = [
                [biases_layers for l in range(0, self.L[m], 1)] for m in range(0, self.M, 1)]  # type: list
        elif isinstance(biases_layers, (list, tuple, np.ndarray, torch.Tensor)):
            tmp_M = len(biases_layers)
            if (tmp_M == self.M) or (tmp_M == 1):
                index_m = 0
                self.biases_layers = [None for m in range(0, self.M, 1)]  # type: list
                for m in range(0, self.M, 1):
                    if tmp_M == self.M:
                        index_m = m
                    if isinstance(biases_layers[index_m], bool):
                        self.biases_layers[m] = [biases_layers[index_m] for l in range(0, self.L[m], 1)]
                    elif isinstance(biases_layers[m], (list, tuple, np.ndarray, torch.Tensor)):
                        tmp_len_biases_layers_m = len(biases_layers[m])
                        if tmp_len_biases_layers_m == self.L[m]:
                            if isinstance(biases_layers[index_m], list):
                                self.biases_layers[m] = biases_layers[index_m]
                            elif isinstance(biases_layers[index_m], tuple):
                                self.biases_layers[m] = list(biases_layers[index_m])
                            elif isinstance(biases_layers[index_m], (np.ndarray, torch.Tensor)):
                                self.biases_layers[m] = biases_layers[index_m].tolist()
                        elif tmp_len_biases_layers_m == 1:
                            if isinstance(biases_layers[index_m], (list, tuple)):
                                biases_layers_m = biases_layers[index_m][0]
                            elif isinstance(biases_layers[index_m], (np.ndarray, torch.Tensor)):
                                biases_layers_m = biases_layers[index_m][0].tolist()
                            else:
                                raise TypeError()
                            self.biases_layers[m] = [biases_layers_m for l in range(0, self.L[m], 1)]
                        else:
                            raise ValueError('len(biases_layers[' + str(m) + '])')
                    else:
                        raise TypeError('biases_layers[' + str(m) + ']')
            else:
                raise ValueError('biases_layers = ' + str(biases_layers))

        else:
            raise TypeError('biases_layers')

        if axis_features is None:
            self.axis_features = -1
        elif isinstance(axis_features, int):
            self.axis_features = axis_features
        else:
            raise TypeError('axis_features')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
