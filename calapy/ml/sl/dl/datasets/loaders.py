# websites:
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/hub/pytorch_vision_resnet/
# https://discuss.pytorch.org/t/normalize-each-input-image-in-a-batch-independently-and-inverse-normalize-the-output/23739
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


import torch
import torchvision
import PIL
import numpy as np
import math
import os
import copy
import typing
import types
from ..datasets import tools as cp_tools
from ..devices import define_device as cp_define_device
from ..... import array as cp_array
from ..... import txt as cp_txt
from ..... import directory as cp_directory
from ..... import maths as cp_maths
from ..... import combinations as cp_combinations

__all__ = ['BatchLoader', 'FileLoader']


class FileLoader:

    # todo: this code may have bugs

    def __init__(
            self, format_file: str = None, directory_file: str = None,
            indexes: typing.Optional[tuple] = None,
            transforms: typing.Union[torchvision.transforms.Compose, None] = None,
            device: typing.Union[torch.device, str, None] = None):

        self.format_file = cp_tools.define_file_format(format_file=format_file, directory_file=directory_file)

        if self.format_file in ['png']:
            self.load_type = 'image'
        elif self.format_file in ['csv']:
            self.load_type = 'csv'
        else:
            raise ValueError('format_file')

        if self.load_type == 'image':

            self.load_file = self.load_image

            self.transforms = cp_tools.define_transforms(transforms=transforms)

        elif self.load_type == 'csv':
            self.load_file = self.load_csv
            # self.transforms = None

        self.indexes = indexes
        if self.indexes is None:
            self.load = self.load_without_indexes
        else:
            self.load = self.load_with_indexes

        self.device = cp_define_device(device=device)

    def load_with_indexes(self, directory_file: str):

        data_pt = self.load_file(directory_file)[self.indexes]

        return data_pt

    def load_without_indexes(self, directory_file: str):

        data_pt = self.load_file(directory_file)

        return data_pt

    def load_image(self, directory_file: str):

        data_pt = self.transforms(PIL.Image.open(directory_file)).to(device=self.device)

        return data_pt

    def load_csv(self, directory_file: str):
        # todo: define rows and columns
        data_np = cp_txt.csv_file_to_array(directory_file, rows=None, columns=None, dtype='f')

        data_pt = torch.tensor(data_np, dtype=torch.float32, device=self.device)

        return data_pt

    # self.__call__ = self.load
    def __call__(self, directory_file: str):

        data_pt = self.load(directory_file=directory_file)

        return data_pt


class BatchLoader:
    """
    A class used to load batches of samples

    Extended description

    Attributes
    ----------
    directory_root : str
        A formatted string to print out what the animal says
    shifts : Shifts
        The name of the animal
    L, n_levels_directories : int
        The number of the directory tree in the dataset.
    conditions_directories : sequence of sequences of ints
        The l_th element of the sequence contains the conditions of the l_th directory level that will be loaded

    Methods
    -------
    load_batches_e(order_outputs=None)
        It is a batch generator. It load a batch of samples at the time.
    """

    # TODO at the moment assumes that all dimensions of each file are within-sample or intra-sample.
    #  Add optional feature that allows loading the files that has one or more dimensions between-samples or
    #  inter-sample. In other words add file_axes_inter and file_axes_intra

    def __init__(
            self, directory_root, format_file=None,
            conditions_directories=None, levels_labels=None,
            levels_inter=None, levels_intra=None,
            levels_dynamics=None, func_dynamic_inter=None, func_dynamic_intra=None,
            T: typing.Union[int, float, None] = None,
            file_axes_inputs=None, batch_axis_inputs=None, unsqueeze_axes_inputs=None,
            axes_inputs_in_labels: typing.Union[tuple, list, np.ndarray, int, None] = None,
            axes_inputs_in_combinations_inter: typing.Union[tuple, list, np.ndarray, int, None] = None,
            indexes: typing.Union[tuple, list, np.ndarray, slice, int, None] = None,
            transforms: typing.Union[torchvision.transforms.Compose, None] = None,
            batch_size=None, n_batches=None, shuffle=False, shifts=None,
            device: typing.Union[torch.device, str, None] = None, order_outputs='il'):

        """
        Parameters
        ----------
        directory_root : str
            The directory of the dataset
        conditions_directories : None or sequence of Nones and sequences of ints, optional
            Conditions_directories can be a sequence or None (default is None). If it is a sequence, the l_th element
            of it is None or a sequence of ints. If the l_th element is a sequence of ints, it contains the conditions
            of the l_th level that will be loaded. If the l_th element is None all conditions of the l_th level
            will be loaded. If conditions_directories is None all conditions in all level will be loaded.
        levels_labels : int or sequence of ints, optional
            The directory levels of the classes in the directory tree of the dataset.
        order_outputs : str or sequence of str, optional
            The desired outputs of the self.load_batches_e(). Accepted values are "i", "l", "c","r", "a" or any
            combination of them like "ilcr", "ilr" (default is "il"). "i" stands for input samples, "l" for labels of
            the samples, "c" for combinations of the level conditions of the samples, "r" for the relative directories
            of the samples, "a" for the absolute directories of the samples.
        """

        self.directory_root = directory_root

        l = 0
        self.conditions_directories_names = []
        directory_root_l = self.directory_root
        while os.path.isdir(directory_root_l):
            self.conditions_directories_names.append(sorted(os.listdir(directory_root_l)))
            directory_root_l = os.path.join(directory_root_l, self.conditions_directories_names[l][0])
            l += 1
        self.L = self.n_levels_directories = l
        del l, directory_root_l
        self.levels_all = np.arange(self.L, dtype='i')

        if levels_labels is None:
            self.levels_labels = np.asarray([], dtype='i')
        elif isinstance(levels_labels, (list, tuple)):
            self.levels_labels = np.asarray(levels_labels, dtype='i')
        elif isinstance(levels_labels, np.ndarray):
            if levels_labels.ndim == 0:
                self.levels_labels = np.expand_dims(levels_labels, axis=0)
            elif levels_labels.ndim == 1:
                self.levels_labels = levels_labels
            else:
                raise ValueError('levels_labels')
        elif isinstance(levels_labels, int):
            self.levels_labels = np.asarray([levels_labels], dtype='i')
        else:
            raise TypeError('levels_labels')
        self.n_levels_labels = self.C = len(self.levels_labels)
        self.levels_labels[self.levels_labels < 0] += self.L

        if levels_intra is None:
            self.levels_intra = np.asarray([], dtype='i')
        elif isinstance(levels_intra, (list, tuple)):
            self.levels_intra = np.asarray(levels_intra, dtype='i')
        elif isinstance(levels_intra, np.ndarray):
            if levels_intra.ndim == 0:
                self.levels_intra = np.expand_dims(levels_intra, axis=0)
            elif levels_intra.ndim == 1:
                self.levels_intra = levels_intra
            else:
                raise ValueError('levels_intra')
        elif isinstance(levels_intra, int):
            self.levels_intra = np.asarray([levels_intra], dtype='i')
        else:
            raise TypeError('levels_intra')
        self.H = self.n_levels_directories_intra = len(self.levels_intra)

        self.levels_intra[self.levels_intra < 0] += self.L
        self.levels_intra_sort = np.sort(self.levels_intra, axis=0)

        if levels_dynamics is None:
            self.levels_dynamics = np.asarray([], dtype='i')
        elif isinstance(levels_dynamics, (list, tuple)):
            self.levels_dynamics = np.asarray(levels_dynamics, dtype='i')
        elif isinstance(levels_dynamics, np.ndarray):
            if levels_dynamics.ndim == 0:
                self.levels_dynamics = np.expand_dims(levels_dynamics, axis=0)
            elif levels_dynamics.ndim == 1:
                self.levels_dynamics = levels_dynamics
            else:
                raise ValueError('levels_dynamics')
        elif isinstance(levels_dynamics, int):
            self.levels_dynamics = np.asarray([levels_dynamics], dtype='i')
        else:
            raise TypeError('levels_dynamics')
        self.n_levels_dynamics = self.D = len(self.levels_dynamics)
        self.levels_dynamics[self.levels_dynamics < 0] += self.L

        if self.n_levels_dynamics > 0:
            if T is None:
                self.T = math.inf
            elif isinstance(T, (int, float)):
                self.T = T
            else:
                raise TypeError(T)

        if levels_inter is None:
            non_levels_inter = self.levels_intra
            self.levels_inter = (
                self.levels_all[cp_array.samples_in_arr1_are_not_in_arr2(self.levels_all, non_levels_inter)])
        elif isinstance(levels_inter, (list, tuple)):
            self.levels_inter = np.asarray(levels_inter, dtype='i')
        elif isinstance(levels_inter, np.ndarray):
            if levels_inter.ndim == 0:
                self.levels_inter = np.expand_dims(levels_inter, axis=0)
            elif levels_inter.ndim == 1:
                self.levels_inter = levels_inter
            else:
                raise ValueError('levels_inter')
        elif isinstance(levels_inter, int):
            self.levels_inter = np.asarray([levels_inter], dtype='i')
        else:
            raise TypeError('levels_inter')
        self.G = self.n_levels_directories_inter = len(self.levels_inter)

        self.levels_inter[self.levels_inter < 0] += self.L
        self.levels_inter_sort = np.sort(self.levels_inter, axis=0)

        if any(cp_array.samples_in_arr1_are_not_in_arr2(self.levels_inter, self.levels_all)):
            raise ValueError('levels_inter')

        if any(cp_array.samples_in_arr1_are_not_in_arr2(self.levels_intra, self.levels_all)):
            raise ValueError('levels_intra')

        if any(cp_array.samples_in_arr1_are_in_arr2(self.levels_inter, self.levels_intra)):
            raise ValueError('levels_inter, levels_intra')

        levels_all_selected = np.concatenate([self.levels_inter, self.levels_intra], axis=0)
        if any(cp_array.samples_in_arr1_are_not_in_arr2(self.levels_all, levels_all_selected)):
            raise ValueError('levels_intra, levels_inter')

        if any(cp_array.samples_in_arr1_are_not_in_arr2(self.levels_dynamics, self.levels_all)):
            raise ValueError('levels_dynamics')

        if any(cp_array.samples_in_arr1_are_not_in_arr2(self.levels_labels, self.levels_inter)):
            raise ValueError('levels_labels')

        self.variables_labels_in_inter = np.empty(self.n_levels_labels, dtype='i')
        for l in range(0, self.n_levels_labels, 1):
            self.variables_labels_in_inter[l] = np.where(
                self.levels_labels[l] == self.levels_inter)[0][0]

        # self.variables_dynamics_in_inter = np.empty(self.n_levels_dynamics, dtype='i')
        self.variables_dynamics_in_inter = []
        self.variables_dynamics_in_intra = []

        self.variables_inter_in_dynamics_sort = []
        self.variables_intra_in_dynamics_sort = []
        for d in range(0, self.n_levels_dynamics, 1):
            if self.levels_dynamics[d] in self.levels_inter:
                self.variables_dynamics_in_inter.append(
                    np.where(self.levels_dynamics[d] == self.levels_inter)[0][0].tolist())

                self.variables_inter_in_dynamics_sort.append(d)
                # self.variables_inter_in_dynamics.append(np.where(
                #     self.levels_inter[self.variables_dynamics_in_inter[-1]] ==
                #     self.levels_dynamics)[0][0].tolist())

            elif self.levels_dynamics[d] in self.levels_intra:
                self.variables_dynamics_in_intra.append(
                    np.where(self.levels_dynamics[d] == self.levels_intra)[0][0].tolist())

                self.variables_intra_in_dynamics_sort.append(d)
                # self.variables_intra_in_dynamics.append(np.where(
                #     self.levels_intra[self.variables_dynamics_in_intra[-1]] ==
                #     self.levels_dynamics)[0][0].tolist())
            else:
                raise ValueError('levels_dynamics')

        self.variables_inter_in_dynamics = []
        for g in range(0, self.G, 1):
            if self.levels_inter[g] in self.levels_dynamics:
                self.variables_inter_in_dynamics.append(
                    np.where(self.levels_inter[g] == self.levels_dynamics)[0][0].tolist())

        self.variables_intra_in_dynamics = []
        for h in range(0, self.H, 1):
            if self.levels_intra[h] in self.levels_dynamics:
                self.variables_intra_in_dynamics.append(
                    np.where(self.levels_intra[h] == self.levels_dynamics)[0][0].tolist())

        self.n_dynamics_in_inter = len(self.variables_dynamics_in_inter)
        self.n_dynamics_in_intra = len(self.variables_dynamics_in_intra)

        if conditions_directories is None:
            self.conditions_directories = [None for l in range(0, self.L, 1)]  # type: list
        else:
            try:
                conditions_directories[0]
            except TypeError:
                raise TypeError('conditions_directories')
            if len(conditions_directories) == self.L:
                self.conditions_directories = conditions_directories
            else:
                raise ValueError('conditions_directories')

        self.n_conditions_directories = np.empty(self.L, dtype='i')

        for l in range(self.L):
            if self.conditions_directories[l] is None:
                self.n_conditions_directories[l] = len(self.conditions_directories_names[l])
                self.conditions_directories[l] = np.arange(self.n_conditions_directories[l])
            else:
                self.n_conditions_directories[l] = len(self.conditions_directories[l])

                if isinstance(self.conditions_directories[l], int):
                    self.conditions_directories[l] = np.asarray([self.conditions_directories[l]])
                elif isinstance(self.conditions_directories[l], (list, tuple)):
                    self.conditions_directories[l] = np.asarray(self.conditions_directories[l])
                elif isinstance(self.conditions_directories[l], np.ndarray):
                    if self.conditions_directories[l].ndim == 0:
                        self.conditions_directories[l] = np.expand_dims(self.conditions_directories[l], axis=0)
                    elif self.conditions_directories[l].ndim > 1:
                        raise ValueError('conditions_directories[l]')
                else:
                    raise TypeError

        if (self.n_dynamics_in_inter == 0) or (func_dynamic_inter is None):
            self.func_dynamic_inter = None
        else:
            self.func_dynamic_inter = func_dynamic_inter
            if self.func_dynamic_inter.n_conditions is None:
                self.func_dynamic_inter.set_n_conditions(
                    [len(self.conditions_directories_names[d]) for d in self.levels_dynamics if d in self.levels_inter])

        if (self.n_dynamics_in_intra == 0) or (func_dynamic_intra is None):
            self.func_dynamic_intra = None
        else:
            self.func_dynamic_intra = func_dynamic_intra
            if self.func_dynamic_intra.n_conditions is None:
                self.func_dynamic_intra.set_n_conditions(
                    [len(self.conditions_directories_names[d]) for d in self.levels_dynamics if d in self.levels_intra])

        self.conditions_directories_names_inter = [None for g in range(0, self.G, 1)]  # type: list
        self.n_conditions_directories_inter = np.empty(self.G, dtype='i')
        self.conditions_directories_inter = [None for g in range(0, self.G, 1)]  # type: list
        g = 0
        for l in self.levels_inter:
            self.conditions_directories_names_inter[g] = self.conditions_directories_names[l]
            self.n_conditions_directories_inter[g] = self.n_conditions_directories[l]
            self.conditions_directories_inter[g] = self.conditions_directories[l]
            g += 1

        self.conditions_directories_names_intra = [None for h in range(0, self.H, 1)]  # type: list
        self.n_conditions_directories_intra = np.empty(self.H, dtype='i')
        self.conditions_directories_intra = [None for h in range(0, self.H, 1)]  # type: list
        h = 0
        for l in self.levels_intra:
            self.conditions_directories_names_intra[h] = self.conditions_directories_names[l]
            self.n_conditions_directories_intra[h] = self.n_conditions_directories[l]
            self.conditions_directories_intra[h] = self.conditions_directories[l]
            h += 1

        if self.n_levels_labels == 0:
            self.K = self.n_classes = None
        else:
            self.K = self.n_classes = self.n_conditions_directories[self.levels_labels]

        self.n_samples = cp_maths.prod(self.n_conditions_directories_inter)
        # self.n_samples = math.prod(self.n_conditions_directories)

        self.indexes = indexes
        self.transforms = cp_tools.define_transforms(transforms=transforms)

        if batch_size is None:
            if n_batches is None:
                self.batch_size = self.n_samples
                self.n_batches = 1
            else:
                self.n_batches = n_batches
                self.batch_size = math.floor(self.n_samples / self.n_batches)
        else:
            self.batch_size = batch_size
            self.n_batches = math.floor(self.n_samples / self.batch_size)

        self.n_samples_e = self.n_batches * self.batch_size
        if self.n_samples_e < self.n_samples:
            print('Warming: self.n_samples_e < self.n_samples')

        self.shuffle = shuffle

        self.size_bin_indexes_samples = self.n_samples - self.n_samples_e

        if self.shuffle:
            self.indexes_samples = np.random.permutation(np.arange(self.n_samples))
        else:
            self.indexes_samples = np.arange(self.n_samples)

        indexes_samples_e, self.bin_indexes_samples = np.split(self.indexes_samples, [self.n_samples_e], axis=0)
        self.size_bin_indexes_samples = len(self.bin_indexes_samples)

        if (not self.shuffle) and (self.size_bin_indexes_samples == 0):
            self.batches_indexes = np.split(indexes_samples_e, self.n_batches, axis=0)
        else:
            self.batches_indexes = None

        self.shifts = shifts  # type: cp_tools.Shifts

        self.device = cp_define_device(device=device)

        self.order_accepted_values = 'ilcra'
        if order_outputs is None:
            self.order_outputs = 'il'
            self.n_outputs = 2
        else:
            self.order_outputs = order_outputs
            self.n_outputs = len(self.order_outputs)
            for o in range(self.n_outputs):
                if not (self.order_outputs[o] in self.order_accepted_values):
                    raise ValueError('order_outputs')

        # self.outputs = [None for o in range(0, self.n_outputs, 1)]  # type: list

        self.return_inputs_eb = 'i' in self.order_outputs
        self.return_labels_eb = 'l' in self.order_outputs
        self.return_combinations_eb = 'c' in self.order_outputs
        self.return_relative_directories_eb = 'r' in self.order_outputs
        self.return_absolute_directories_eb = 'a' in self.order_outputs

        print('self.batch_size =', self.batch_size)
        print('self.n_batches =', self.n_batches)
        print('self.n_samples_e =', self.n_samples_e)
        print('self.n_samples =', self.n_samples)
        print('self.shifts =', self.shifts)
        print('self.n_conditions_directories =', self.n_conditions_directories)
        print('self.conditions_directories =', self.conditions_directories)
        print('self.conditions_directories_names =', self.conditions_directories_names)

        self.levels_shifts_inter = []
        self.ranges_shifts_inter = []
        self.levels_shifts_inter_sort = []

        self.levels_shifts_intra = []
        self.ranges_shifts_intra = []
        self.levels_shifts_intra_sort = []

        if self.shifts is None:
            self.shifts_inter = None
            self.shifts_intra = None
        else:
            self.shifts.levels[self.shifts.levels < 0] += self.L

            for v in range(self.shifts.n_levels):
                if self.shifts.levels[v] in self.levels_inter:
                    self.levels_shifts_inter.append(np.where(self.shifts.levels[v] == self.levels_inter)[0][0].tolist())
                    self.ranges_shifts_inter.append(self.shifts.ranges[v])
                elif self.shifts.levels[v] in self.levels_intra:
                    self.levels_shifts_intra.append(np.where(self.shifts.levels[v] == self.levels_intra)[0][0].tolist())
                    self.ranges_shifts_intra.append(self.shifts.ranges[v])
                else:
                    raise ValueError('levels_inter, levels_intra')

            if len(self.levels_shifts_inter) == 0:
                self.shifts_inter = None
            else:
                self.shifts_inter = cp_tools.Shifts(
                    self.ranges_shifts_inter, self.levels_shifts_inter, self.n_samples_e)
                self.levels_shifts_inter = self.shifts_inter.levels
                self.levels_shifts_inter_sort = self.shifts_inter.levels_sort
                self.ranges_shifts_inter = self.shifts_inter.ranges

            if len(self.levels_shifts_intra) == 0:
                self.shifts_intra = None
            else:
                self.shifts_intra = cp_tools.Shifts(
                    self.ranges_shifts_intra, self.levels_shifts_intra, self.n_samples_e)
                self.levels_shifts_intra = self.shifts_intra.levels
                self.levels_shifts_intra_sort = self.shifts_intra.levels_sort
                self.ranges_shifts_intra = self.shifts_intra.ranges

        self.batches_shifts_intra = None
        self.shifts_intra_eb = None

        if self.shifts_inter is not None:
            self.combinations_directories_inter_no_shift = (
                cp_combinations.conditions_to_combinations(self.conditions_directories_inter))
            self.combinations_directories_inter = None
            self.labels = None

        else:
            self.combinations_directories_inter_no_shift = None
            self.combinations_directories_inter = (
                cp_combinations.conditions_to_combinations(self.conditions_directories_inter))
            if self.return_labels_eb:

                self.labels = [torch.tensor(
                    self.combinations_directories_inter[slice(0, self.n_samples, 1), index],
                    dtype=torch.int64, device=self.device, requires_grad=False)
                    for index in self.variables_labels_in_inter]
            else:
                self.labels = None

        self.combinations_inter_eb = None
        self.combination_inter_ebi = None

        # self.labels_eb = None

        self.combinations_directories_intra = (
            cp_combinations.conditions_to_combinations(self.conditions_directories_intra))

        if self.shifts_intra is not None:
            self.combinations_intra_eb = [None for i in range(0, self.batch_size, 1)]
        else:
            self.combinations_intra_eb = (
                [self.combinations_directories_intra for i in range(0, self.batch_size, 1)])

        self.combination_intra_ebij = None

        self.combinations_indexes_input_intra = (
            cp_combinations.n_conditions_to_combinations(self.n_conditions_directories_intra))

        combination_directory_str_0 = [self.conditions_directories_names[l][0] for l in range(self.L)]
        directory_0 = os.path.join(self.directory_root, *combination_directory_str_0)

        if format_file is None:
            self.format_file = cp_directory.get_extension(directory_0, point=False).lower()
        elif isinstance(format_file, str):
            if format_file[0] == '.':
                self.format_file = format_file[1:].lower()
            else:
                self.format_file = format_file.lower()
        else:
            raise TypeError('format_file')

        self.file_loader = FileLoader(
            format_file=self.format_file, directory_file=directory_0,
            indexes=self.indexes, transforms=self.transforms, device=self.device)

        # array_np_0 = cp_txt.csv_file_to_array(directory_0, rows=self.rows, columns=self.columns, dtype='f')
        # tensor_0 = torch.tensor(array_np_0, dtype=torch.float32, device=self.device)

        tensor_0 = self.file_loader(directory_0)

        shape_file_0 = list(tensor_0.shape)
        self.shape_file_data = shape_file_0
        self.n_dims_file_data = self.shape_file_data.__len__()

        if unsqueeze_axes_inputs is None:
            self.unsqueeze_axes_inputs = np.empty([0], dtype='i')
        elif isinstance(unsqueeze_axes_inputs, int):
            self.unsqueeze_axes_inputs = np.asarray([unsqueeze_axes_inputs], dtype='i')
        elif isinstance(unsqueeze_axes_inputs, (list, tuple)):
            self.unsqueeze_axes_inputs = np.asarray(unsqueeze_axes_inputs, dtype='i')
        elif isinstance(unsqueeze_axes_inputs, np.ndarray):
            if unsqueeze_axes_inputs.ndim == 0:
                self.unsqueeze_axes_inputs = np.expand_dims(unsqueeze_axes_inputs, axis=0)
            elif unsqueeze_axes_inputs.ndim == 1:
                self.unsqueeze_axes_inputs = unsqueeze_axes_inputs
            else:
                raise ValueError('unsqueeze_axes_inputs')
        else:
            raise TypeError('unsqueeze_axes_inputs')

        self.n_unsqueezes = len(self.unsqueeze_axes_inputs)

        self.n_dims_samples = self.n_dims_file_data + self.H + self.n_unsqueezes
        self.n_dims_batch = self.n_dims_samples + 1
        self.n_dims_directories_eb = self.H + 1 + self.n_unsqueezes

        self.unsqueeze_axes_inputs[self.unsqueeze_axes_inputs < 0] += self.n_dims_batch

        self.axes_inputs = np.arange(0, self.n_dims_batch, 1, dtype='i')

        if file_axes_inputs is None:
            self.file_axes_inputs = np.arange(
                self.n_dims_batch - self.n_dims_file_data, self.n_dims_batch, 1, dtype='i')
        elif isinstance(file_axes_inputs, int):
            self.file_axes_inputs = np.asarray([file_axes_inputs], dtype='i')
        elif isinstance(file_axes_inputs, (list, tuple)):
            self.file_axes_inputs = np.asarray(file_axes_inputs, dtype='i')
        elif isinstance(file_axes_inputs, np.ndarray):
            if file_axes_inputs.ndim == 0:
                self.file_axes_inputs = np.expand_dims(file_axes_inputs, axis=0)
            elif file_axes_inputs.ndim == 1:
                self.file_axes_inputs = file_axes_inputs
            else:
                raise ValueError('file_axes_inputs')
        else:
            raise TypeError('file_axes_inputs')

        self.file_axes_inputs[self.file_axes_inputs < 0] += self.n_dims_batch

        for i in range(0, self.n_unsqueezes, 1):
            if self.unsqueeze_axes_inputs[i] in self.file_axes_inputs:
                if file_axes_inputs is None:
                    self.file_axes_inputs[self.file_axes_inputs <= self.unsqueeze_axes_inputs[i]] -= 1
                else:
                    raise ValueError('file_axes_inputs, unsqueeze_axes_inputs')

        if batch_axis_inputs is None:
            self.batch_axis_inputs = 0
            defined_axes_inputs = self.file_axes_inputs.tolist() + self.unsqueeze_axes_inputs.tolist()
            while self.batch_axis_inputs in defined_axes_inputs:
                self.batch_axis_inputs += 1
        else:
            self.batch_axis_inputs = batch_axis_inputs
            if self.batch_axis_inputs < 0:
                self.batch_axis_inputs += self.n_dims_batch

            if self.batch_axis_inputs in self.file_axes_inputs:
                if file_axes_inputs is None:
                    self.file_axes_inputs[self.file_axes_inputs <= self.batch_axis_inputs] -= 1
                else:
                    raise ValueError('batch_axis_inputs, file_axes_inputs')
            elif self.batch_axis_inputs in self.unsqueeze_axes_inputs:
                raise ValueError('batch_axis_inputs, unsqueeze_axes_inputs')

        non_intra_axes_inputs = (
                [self.batch_axis_inputs] + self.file_axes_inputs.tolist() + self.unsqueeze_axes_inputs.tolist())

        self.intra_axes_inputs = np.empty(self.H, dtype='i')

        if self.H > 0:

            self.intra_axes_inputs[0] = 0

            while self.intra_axes_inputs[0] in non_intra_axes_inputs:
                self.intra_axes_inputs[0] += 1

        for h in range(1, self.H):
            self.intra_axes_inputs[h] = self.intra_axes_inputs[h - 1] + 1
            while self.intra_axes_inputs[h] in non_intra_axes_inputs:
                self.intra_axes_inputs[h] += 1

        non_sample_axes_input = [self.batch_axis_inputs]

        self.sample_axes_input = self.axes_inputs[cp_array.samples_in_arr1_are_not_in_arr2(
            self.axes_inputs, non_sample_axes_input)]

        self.shape_batch = np.empty(self.n_dims_batch, dtype='i')
        self.shape_batch[self.batch_axis_inputs] = self.batch_size
        self.shape_batch[self.intra_axes_inputs] = self.n_conditions_directories_intra
        self.shape_batch[self.file_axes_inputs] = self.shape_file_data
        self.shape_batch[self.unsqueeze_axes_inputs] = 1

        if self.return_inputs_eb:
            # self.inputs_eb = torch.empty(tuple(self.shape_batch.tolist()), dtype=torch.float32, device=self.device)
            self.indexes_batch = np.empty(self.n_dims_batch, dtype='O')
            for d in range(0, self.n_dims_batch, 1):
                self.indexes_batch[d] = slice(0, self.shape_batch[d], 1)
            self.indexes_batch[self.unsqueeze_axes_inputs] = 0
        else:
            # self.inputs_eb = None
            self.indexes_batch = None

        if axes_inputs_in_labels is None:
            self.axes_inputs_in_labels = np.asarray([self.batch_axis_inputs], dtype='i')
        elif isinstance(axes_inputs_in_labels, int):
            self.axes_inputs_in_labels = np.asarray([axes_inputs_in_labels], dtype='i')
        elif isinstance(axes_inputs_in_labels, (list, tuple)):
            self.axes_inputs_in_labels = np.asarray(axes_inputs_in_labels, dtype='i')
        elif isinstance(axes_inputs_in_labels, np.ndarray):
            if axes_inputs_in_labels.ndim == 0:
                self.axes_inputs_in_labels = np.expand_dims(axes_inputs_in_labels, axis=0)
            elif axes_inputs_in_labels.ndim == 1:
                self.axes_inputs_in_labels = axes_inputs_in_labels
            else:
                raise ValueError('axes_inputs_in_labels')
        else:
            raise TypeError('axes_inputs_in_labels')

        self.axes_inputs_in_labels[self.axes_inputs_in_labels < 0] += self.n_dims_batch

        if self.batch_axis_inputs not in self.axes_inputs_in_labels:
            raise ValueError('axes_inputs_in_labels')

        for a in self.axes_inputs_in_labels:
            if sum(a == self.axes_inputs_in_labels) > 1:
                raise ValueError('axes_inputs_in_labels')

        self.n_dims_labels = 1 + len(self.axes_inputs_in_labels)
        self.axes_labels = np.arange(0, self.n_dims_labels, 1, dtype='i')
        self.model_axis_labels = 0
        self.batch_axis_labels = np.where(self.axes_inputs_in_labels == self.batch_axis_inputs)[0][0].tolist()
        if self.batch_axis_labels >= self.model_axis_labels:
            self.batch_axis_labels += 1

        self.other_axes_labels = [
            a for a in self.axes_labels if a not in [self.model_axis_labels, self.batch_axis_labels]]
        # self.other_axes_labels_list = self.other_axes_labels.tolist()
        self.shape_labels = np.insert(
            self.shape_batch[self.axes_inputs_in_labels], self.model_axis_labels, self.C, axis=0)
        self.shape_labels_list = self.shape_labels.tolist()

        # -----------------------------------------------------------
        if axes_inputs_in_combinations_inter is None:
            self.axes_inputs_in_combinations_inter = np.asarray([self.batch_axis_inputs], dtype='i')
        elif isinstance(axes_inputs_in_combinations_inter, int):
            self.axes_inputs_in_combinations_inter = np.asarray([axes_inputs_in_combinations_inter], dtype='i')
        elif isinstance(axes_inputs_in_combinations_inter, (list, tuple)):
            self.axes_inputs_in_combinations_inter = np.asarray(axes_inputs_in_combinations_inter, dtype='i')
        elif isinstance(axes_inputs_in_combinations_inter, np.ndarray):
            if axes_inputs_in_combinations_inter.ndim == 0:
                self.axes_inputs_in_combinations_inter = np.expand_dims(axes_inputs_in_combinations_inter, axis=0)
            elif axes_inputs_in_combinations_inter.ndim == 1:
                self.axes_inputs_in_combinations_inter = axes_inputs_in_combinations_inter
            else:
                raise ValueError('axes_inputs_in_combinations_inter')
        else:
            raise TypeError('axes_inputs_in_combinations_inter')

        self.axes_inputs_in_combinations_inter[self.axes_inputs_in_combinations_inter < 0] += self.n_dims_batch

        if self.batch_axis_inputs not in self.axes_inputs_in_combinations_inter:
            raise ValueError('axes_inputs_in_combinations_inter')

        for a in self.axes_inputs_in_combinations_inter:
            if sum(a == self.axes_inputs_in_combinations_inter) > 1:
                raise ValueError('axes_inputs_in_combinations_inter')

        self.n_dims_combinations_inter = 1 + len(self.axes_inputs_in_combinations_inter)
        self.axes_combinations_inter = np.arange(0, self.n_dims_combinations_inter, 1, dtype='i')
        self.variable_axis_combinations_inter = self.n_dims_combinations_inter - 1
        self.batch_axis_combinations_inter = np.where(self.axes_inputs_in_combinations_inter == self.batch_axis_inputs)[0][0].tolist()
        if self.batch_axis_combinations_inter >= self.variable_axis_combinations_inter:
            self.batch_axis_combinations_inter += 1

        self.other_axes_combinations_inter = [
            a for a in self.axes_combinations_inter
            if a not in [self.variable_axis_combinations_inter, self.batch_axis_combinations_inter]]
        # self.other_axes_combinations_inter_list = self.other_axes_combinations_inter.tolist()

        self.shape_combinations_inter = np.insert(
            self.shape_batch[self.axes_inputs_in_combinations_inter],
            self.variable_axis_combinations_inter, self.G, axis=0)
        self.shape_combinations_inter_list = self.shape_combinations_inter.tolist()

        self.intra_axes_directories = []
        self.batch_axis_directories = None
        self.unsqueeze_axes_directories = []

        f = 0
        for i in self.axes_inputs:
            if i in self.file_axes_inputs:
                f += 1
            elif i == self.batch_axis_inputs:
                self.batch_axis_directories = i - f
            elif i in self.intra_axes_inputs:
                self.intra_axes_directories.append(i - f)
            elif i == self.unsqueeze_axes_inputs:
                self.unsqueeze_axes_directories.append(i - f)

        self.intra_axes_directories = np.asarray(self.intra_axes_directories, dtype='i')
        self.unsqueeze_axes_directories = np.asarray(self.unsqueeze_axes_directories, dtype='i')

        self.shape_directories_eb = np.empty(self.n_dims_directories_eb, dtype='i')
        self.shape_directories_eb[self.intra_axes_directories] = self.shape_batch[self.intra_axes_inputs]
        self.shape_directories_eb[self.batch_axis_directories] = self.shape_batch[self.batch_axis_inputs]
        self.shape_directories_eb[self.unsqueeze_axes_directories] = 1

        if self.return_relative_directories_eb or self.return_absolute_directories_eb:
            self.indexes_directories_ebij = np.empty(self.n_dims_directories_eb, dtype='O')
            self.indexes_directories_ebij[self.unsqueeze_axes_directories] = 0
        else:
            self.indexes_directories_ebij = None

        self.combinations_ebij = np.empty(self.L, dtype='i')

        self.b = 0
        self.s = 0

        print('self.shape_batch =', self.shape_batch)
        print()

    def __iter__(self):

        if self.shifts_inter is not None:
            self.shifts_inter.refresh()
            # TODO: recopy only the variable conditions with shifts
            self.combinations_directories_inter = np.copy(self.combinations_directories_inter_no_shift)
            self.combinations_directories_inter[:, self.shifts_inter.levels] += self.shifts_inter.values
            if self.return_labels_eb:
                self.labels = [torch.tensor(
                    self.combinations_directories_inter[slice(0, self.n_samples, 1), index],
                    dtype=torch.int64, device=self.device, requires_grad=False)
                    for index in self.variables_labels_in_inter]
            else:
                self.labels = None

        if self.shifts_intra is not None:
            self.shifts_intra.refresh()
            self.batches_shifts_intra = np.split(self.shifts_intra.values, self.n_batches, axis=0)

        if self.shuffle:
            indexes_samples_e, self.bin_indexes_samples = np.split(
                np.append(self.bin_indexes_samples, np.random.permutation(self.indexes_samples), axis=0),
                [self.n_samples_e], axis=0)

            self.size_bin_indexes_samples = len(self.bin_indexes_samples)
            self.batches_indexes = np.split(indexes_samples_e, self.n_batches, axis=0)
        elif self.size_bin_indexes_samples > 0:

            indexes_samples_e, self.bin_indexes_samples = np.split(
                np.append(self.bin_indexes_samples, self.indexes_samples, axis=0),
                [self.n_samples_e], axis=0)

            self.size_bin_indexes_samples = len(self.bin_indexes_samples)
            self.batches_indexes = np.split(indexes_samples_e, self.n_batches, axis=0)

        self.b = -1
        self.s = -1

        return self

    def __next__(self):

        self.b += 1

        # if self.b < 10:
        if self.b < self.n_batches:

            self.s += self.batch_size

            self.combinations_inter_eb = copy.deepcopy(
                self.combinations_directories_inter[self.batches_indexes[self.b], :])

            if self.shifts_intra is not None:
                self.shifts_intra_eb = self.batches_shifts_intra[self.b]
                for i in range(0, self.batch_size, 1):
                    shifts_intra_s = self.shifts_intra_eb[slice(i, i + 1, 1), :]
                    # TODO: recopy only the variable conditions with shifts
                    self.combinations_intra_eb[i] = np.copy(self.combinations_directories_intra)
                    self.combinations_intra_eb[i][:, self.shifts_intra.levels] += shifts_intra_s

            if self.n_levels_dynamics > 0:
                self.stepper = Stepper(self)
                return self.stepper
            else:
                outputs = self()
                return outputs
        else:
            raise StopIteration

    def load(self):
        """

        Returns
        -------

        """

        outputs = [None for o in range(0, self.n_outputs, 1)]  # type: list

        if self.return_inputs_eb:
            inputs_eb = torch.empty(tuple(self.shape_batch.tolist()), dtype=torch.float32, device=self.device)

        combinations_inter_eb = copy.deepcopy(self.combinations_inter_eb)

        if self.return_relative_directories_eb:
            relative_directories_eb = np.empty(self.shape_directories_eb, dtype='O')

        if self.return_absolute_directories_eb:
            absolute_directories_eb = np.empty(self.shape_directories_eb, dtype='O')

        for i in range(0, self.batch_size, 1):

            if self.return_inputs_eb:
                self.indexes_batch[self.batch_axis_inputs] = i

            if self.return_relative_directories_eb or self.return_absolute_directories_eb:
                self.indexes_directories_ebij[self.batch_axis_directories] = i

            self.combination_inter_ebi = combinations_inter_eb[i, :]

            self.combinations_ebij[self.levels_inter] = self.combination_inter_ebi

            if self.combinations_intra_eb[i].shape[0] > 0:

                for j in range(self.combinations_intra_eb[i].shape[0]):

                    self.combination_intra_ebij = self.combinations_intra_eb[i][j, :]

                    self.combinations_ebij[self.levels_intra] = self.combination_intra_ebij

                    combination_directory_str_ebij = [
                        self.conditions_directories_names[l][self.combinations_ebij[l]] for l in range(self.L)]

                    relative_directory_ebij = os.path.join(*combination_directory_str_ebij)
                    absolute_directory_ebij = os.path.join(self.directory_root, relative_directory_ebij)

                    if self.return_inputs_eb:

                        self.indexes_batch[self.intra_axes_inputs] = self.combinations_indexes_input_intra[j, :]

                        tensor_ebij = self.file_loader(absolute_directory_ebij)

                        # todo move dimensions of tensor_ebi
                        #  if file_axes_inputs[0] > file_axes_inputs[1]

                        inputs_eb[tuple(self.indexes_batch)] = tensor_ebij

                    if self.return_relative_directories_eb or self.return_absolute_directories_eb:
                        self.indexes_directories_ebij[self.intra_axes_directories] = (
                            self.combinations_indexes_input_intra[j, :])

                    if self.return_relative_directories_eb:
                        relative_directories_eb[tuple(self.indexes_directories_ebij)] = relative_directory_ebij

                    if self.return_absolute_directories_eb:
                        absolute_directories_eb[tuple(self.indexes_directories_ebij)] = absolute_directory_ebij

            else:
                combination_directory_str_ebij = [
                    self.conditions_directories_names[l][self.combinations_ebij[l]] for l in range(self.L)]

                relative_directory_ebij = os.path.join(*combination_directory_str_ebij)
                absolute_directory_ebij = os.path.join(self.directory_root, relative_directory_ebij)

                if self.return_inputs_eb:

                    tensor_ebij = self.file_loader(absolute_directory_ebij)

                    # todo: move dimensions of tensor_ebi
                    #  if file_axes_inputs[0] > file_axes_inputs[1]

                    inputs_eb[tuple(self.indexes_batch)] = tensor_ebij

                if self.return_relative_directories_eb:
                    relative_directories_eb[tuple(self.indexes_directories_ebij)] = relative_directory_ebij

                if self.return_absolute_directories_eb:
                    absolute_directories_eb[tuple(self.indexes_directories_ebij)] = absolute_directory_ebij

        for o in range(self.n_outputs):
            if self.order_outputs[o] == 'i':
                outputs[o] = inputs_eb
            elif self.order_outputs[o] == 'l':

                if self.n_dims_labels == 2:
                    labels_eb = torch.tensor(
                        combinations_inter_eb[:, self.variables_labels_in_inter].T,
                        dtype=torch.int64, device=self.device, requires_grad=False)
                elif self.n_dims_labels > 2:
                    labels_eb = torch.tensor(
                        np.expand_dims(
                            combinations_inter_eb[:, self.variables_labels_in_inter].T, axis=self.other_axes_labels),
                        dtype=torch.int64, device=self.device, requires_grad=False).expand(self.shape_labels_list)
                else:
                    raise ValueError('axes_inputs_in_labels')

                outputs[o] = labels_eb

            elif self.order_outputs[o] == 'c':

                if self.n_dims_combinations_inter == 2:
                    combinations_inter_eb = torch.tensor(
                        combinations_inter_eb, dtype=torch.int64, device=self.device, requires_grad=False)
                elif self.n_dims_combinations_inter > 2:
                    combinations_inter_eb = torch.tensor(
                        np.expand_dims(combinations_inter_eb, axis=self.other_axes_combinations_inter),
                        dtype=torch.int64, device=self.device, requires_grad=False).expand(
                        self.shape_combinations_inter_list)
                else:
                    raise ValueError('axes_inputs_in_combinations_inter')

                outputs[o] = combinations_inter_eb

            elif self.order_outputs[o] == 'r':
                outputs[o] = relative_directories_eb
            elif self.order_outputs[o] == 'a':
                outputs[o] = absolute_directories_eb

        return outputs

    __call__ = load
    # def __call__(self):
    #     outputs = self.load()
    #     return outputs


class Stepper:

    def __init__(self, batch_loader: BatchLoader):

        self.batch_loader = batch_loader

    def __iter__(self):

        self.t = -1

        return self

    def __next__(self):

        self.t += 1
        if self.t < self.batch_loader.T:
            outputs_ebt = self()
            return outputs_ebt
        else:
            raise StopIteration

    def step(self, delta):

        if len(self.batch_loader.variables_inter_in_dynamics_sort) > 0:

            delta_inter = delta[:, self.batch_loader.variables_inter_in_dynamics_sort]

            new_dynamic_inter = (
                self.batch_loader.combinations_inter_eb[:, self.batch_loader.variables_dynamics_in_inter] + delta_inter)

            if self.batch_loader.func_dynamic_inter is not None:
                new_dynamic_inter = self.batch_loader.func_dynamic_inter(new_dynamic_inter)

            self.batch_loader.combinations_inter_eb[:, self.batch_loader.variables_dynamics_in_inter] = (
                new_dynamic_inter)

        if len(self.batch_loader.variables_intra_in_dynamics_sort) > 0:
            for i in range(0, self.batch_loader.batch_size, 1):

                delta_intra_i = delta[slice(i, i + 1, 1), self.batch_loader.variables_intra_in_dynamics_sort]

                new_dynamic_intra = (
                    self.batch_loader.combinations_intra_eb[i][:, self.batch_loader.variables_dynamics_in_intra] +
                    delta_intra_i)

                if self.batch_loader.func_dynamic_intra is not None:
                    new_dynamic_intra = self.batch_loader.func_dynamic_intra(new_dynamic_intra)

                self.batch_loader.combinations_intra_eb[i][:, self.batch_loader.variables_dynamics_in_intra] = (
                    new_dynamic_intra)

        return None

    def load(self):

        outputs_ebt = self.batch_loader()

        return outputs_ebt

    __call__ = start = load
