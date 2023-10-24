# websites:
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/hub/pytorch_vision_resnet/
# https://discuss.pytorch.org/t/normalize-each-input-image-in-a-batch-independently-and-inverse-normalize-the-output/23739
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


import torchvision
import typing
import math
import numpy as np
from ... import array as cp_array
from ... import directory as cp_directory

__all__ = ['Shifts', 'define_transforms', 'define_file_format']


class Shifts:
    # TODO make range for each directory level (this is possible if I make condition_to_combination compatible for
    #  ranges like this class Time)

    def __init__(self, ranges_raw, levels, n_shifts=None):

        try:
            len(levels)
            if isinstance(levels, np.ndarray):
                self.levels = levels
            else:
                self.levels = np.asarray(levels, dtype='i')

            n_axes_levels = len(self.levels.shape)
            if n_axes_levels > 1:
                raise ValueError('levels')

            self.n_levels = self.levels.size
        except TypeError:
            self.levels = np.asarray([levels], dtype='i')
            self.n_levels = 1

        if self.n_levels != len(ranges_raw):
            raise ValueError('ranges_raw, levels')

        if any(self.levels < 0):
            self.levels_sort = None
        else:
            self.levels_sort = np.sort(self.levels)

        self.ranges = [None] * self.n_levels  # type: list

        self.unique = [None] * self.n_levels  # type: list
        self.n_unique = [None] * self.n_levels  # type: list

        for v in range(self.n_levels):
            if isinstance(ranges_raw[v], cp_array.IntRange):
                self.ranges[v] = ranges_raw[v]
            else:
                self.ranges[v] = cp_array.IntRange(ranges_raw[v])

            if self.ranges[v].len < 1:
                raise ValueError('range_raw')

            # self.n_shifts = n_shifts

            self.unique[v] = self.ranges[v].to_list()
            self.n_unique[v] = self.unique[v].__len__()

        self.n_shifts = n_shifts
        self.__set_n_shifts__(n_shifts)
        self.i = -1

    def __set_n_shifts__(self, n_shifts):

        self.n_shifts = n_shifts
        if self.n_shifts is None:
            self.bin = None
            self.values = None
            self.n_repetitions = None
            self.values_all = None
        else:
            self.n_repetitions = [None] * self.n_levels  # type: list
            self.values_all = [None] * self.n_levels  # type: list

            self.values = np.empty([self.n_shifts, self.n_levels], dtype='i')
            self.bin = [None] * self.n_levels  # type: list
            for v in range(self.n_levels):

                self.n_repetitions[v] = math.ceil(self.n_shifts / self.n_unique[v])
                self.values_all[v] = np.asarray(self.unique[v] * self.n_repetitions[v], dtype='i')

                self.values[slice(0, self.n_shifts, 1), v], self.bin[v] = (
                    np.split(np.random.permutation(self.values_all[v]), [self.n_shifts], axis=0))

    def set_n_shifts(self, n_shifts):
        self.__set_n_shifts__(n_shifts)

    def __iter__(self):
        self.i = -1

        if self.n_shifts is None:
            raise ValueError('self.n_shifts')
        else:
            for v in range(self.n_levels):

                self.values[slice(0, self.n_shifts, 1), v], self.bin[v] = np.split(
                    np.append(self.bin[v], np.random.permutation(self.values_all[v]), axis=0),
                    [self.n_shifts], axis=0)

        return self

    def __next__(self):
        self.i += 1
        if self.i < self.n_shifts:
            self.values_i = self.__getitem__(self.i)
            return self.values_i
        else:
            raise StopIteration

    def refresh(self):
        self.__iter__()
        return self.values

    def __getitem__(self, i):
        self.values_i = self.values[self.i, slice(0, self.n_levels, 1)]
        return self.values_i

    def __len__(self):
        if self.n_shifts is None:
            raise ValueError('self.n_shifts')
        else:
            return self.n_shifts


def define_transforms(transforms: typing.Union[torchvision.transforms.Compose, None] = None):

    if transforms is None:
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    else:
        to_tensor_is_not_in_transforms = True
        for transform_t in transforms.transforms:
            if type(torchvision.transforms.ToTensor()) == type(transform_t):
                to_tensor_is_not_in_transforms = False
                break
        if to_tensor_is_not_in_transforms:
            transforms.transforms.append(torchvision.transforms.ToTensor())
            to_tensor_is_not_in_transforms = True

        return transforms


def define_file_format(format_file: str = None, directory_file: str = None):

    # format_file_out = cp_directory.get_extension(directory_file, point=False).lower()

    if format_file is None:

        if directory_file is None:
            raise ValueError('format_file, directory_file')

        elif isinstance(directory_file, str):
            format_file_out = cp_directory.get_extension(directory_file, point=False).lower()
        else:
            raise TypeError('directory_file')

    elif isinstance(format_file, str):

        if directory_file is None:

            format_file_out = format_file.lower()
        elif isinstance(directory_file, str):

            format_file_out = format_file.lower()

            format_file_tmp = cp_directory.get_extension(directory_file, point=False).lower()

            if format_file_out != format_file_tmp:
                raise ValueError('format_file or directory_file')
        else:
            raise TypeError('directory_file')

    else:
        raise TypeError('format_file')

    return format_file_out
