import numpy as np
import math
import numbers
# from copy import deepcopy
# from . import combinations as cp_combinations
from . import maths as cp_maths
# from .maths import convert_to_int_or_float, rint


def advanced_indexing(indexes_loose):
    def clean_input_of_ix_(indexes_loose):
        n_axes = len(indexes_loose)
        for a in range(n_axes):
            try:
                len(indexes_loose[a])
            except TypeError:
                indexes_loose[a] = np.asarray([indexes_loose[a]], dtype=int)
        return indexes_loose

    indexes_loose = clean_input_of_ix_(indexes_loose)
    indexes = np.ix_(*indexes_loose)
    return indexes


def samples_in_arr1_are_not_in_arr2(arr1, arr2, axis=0):
    return np.logical_not(samples_in_arr1_are_in_arr2(arr1, arr2, axis=axis))


def samples_in_arr1_are_in_arr2(arr1, arr2, axis=0):

    # OLD elements_in_x_are_in_y()

    arr1, arr2 = make_sure_is_iterable(arr1, arr2)

    arr1 = np.array(arr1)
    type_arr1 = arr1.dtype
    if type_arr1 == object:
        arr1 = np.array(list(arr1))

    arr2 = np.array(arr2)
    type_arr2 = arr2.dtype
    if type_arr2 == object:
        arr2 = np.array(list(arr2))

    shape_arr1 = np.array(arr1.shape)
    shape_arr2 = np.array(arr2.shape)

    n_axes_arr1 = len(shape_arr1)
    n_axes_arr2 = len(shape_arr2)
    if n_axes_arr1 != n_axes_arr2:

        if n_axes_arr1 == n_axes_arr2 - 1:

            arr1 = np.expand_dims(arr1, axis=axis)
            shape_arr1 = np.array(arr1.shape)
            n_axes_arr1 = len(shape_arr1)

        else:
            raise Exception('n_axes_arr1 has to be equal to n_axes_arr2\n'
                            'or to n_axes_arr2 - 1.\n'
                            'Now, n_axes_arr1 = {} and n_axes_arr2 = {}.'.format(
                n_axes_arr1, n_axes_arr2))

    axes_arr1_and_arr2 = np.arange(n_axes_arr1)

    if np.all(axis != axes_arr1_and_arr2):
        raise Exception('axis_1 is not in the axes_arr1_and_arr2.\n'
                        'axis_1 has to be equal to either one of the\n'
                        'axes_arr1_and_arr2.\n'
                        'Now, axis_1 = {} and axes_arr1_and_arr2 = {}.\n'
                        'By default, axis_1=0.'.format(
                         axis, axes_arr1_and_arr2))

    n_samples_arr1 = shape_arr1[axis]
    n_samples_arr2 = shape_arr2[axis]

    axes_within_samples = np.argwhere(axes_arr1_and_arr2 != axis)[:, 0]
    # dimensions_within_samples = dimensions_within_samples.astype(int)

    shape_1_sample_arr1 = shape_arr1[axes_within_samples]
    shape_1_sample_arr2 = shape_arr2[axes_within_samples]
    if np.any(shape_1_sample_arr1 != shape_1_sample_arr2):

        raise Exception('shape_1_sample_arr1 has to be equal to shape_1_sample_arr2.\n'
                        'Now,shape_1_sample_arr1 = {} and shape_1_samples_arr2 = {}.'.format(
            shape_1_sample_arr1, shape_1_sample_arr2))

    shape_arr_logical = np.copy(shape_arr2)
    shape_arr_logical = np.insert(shape_arr_logical, axis, shape_arr1[axis], axis=0)
    n_axes_arr_logical = len(shape_arr_logical)
    arr_logical = np.empty(shape_arr_logical, dtype=bool)

    indexes_arr1 = np.full(n_axes_arr1, slice(None))
    indexes_arr_logical = np.full(n_axes_arr_logical, slice(None))

    # indexes_arr1 = np.empty(n_dimensions_arr1, dtype=object)
    # indexes_arr_logical = np.empty(n_dimensions_arr_logical, dtype=object)
    # for i_dimension in range(n_dimensions_arr_logical):
    #     if i_dimension != axis_1:
    #         # indexes_samples_i_window.append(slice(0, n_samples_window))
    #     #     indexes_arr1[i_dimension] = np.arange([0])
    #     # else:
    #         indexes_arr_logical[i_dimension] = np.arange(shape_arr_logical[i_dimension])
    #         if i_dimension < n_dimensions_arr1:
    #             indexes_arr1[i_dimension] = np.arange(shape_arr1[i_dimension])

    for i_sample in range(n_samples_arr1):
        indexes_arr1[axis] = slice(i_sample, i_sample + 1, 1)
        indexes_arr_logical[axis] = i_sample
        arr_logical[tuple(indexes_arr_logical)] = arr1[tuple(indexes_arr1)] == arr2

        # maybe todo: arr_logical[tuple(indexes_arr_logical)] = np.all(arr1[tuple(indexes_arr1)] == arr2)

    elements_of_arr1_in_arr2_logical = np.any(arr_logical, axis=axis + 1)

    return elements_of_arr1_in_arr2_logical


def make_sure_is_iterable(*variables, if_not_iterable_convert_2_type_variable=np.ndarray):
    # list_types_iterable_allowed = [list, tuple, np.ndarray, str]

    # n_types_iterable_allowed = len(list_types_iterable_allowed)
    # if n_types_iterable_allowed == 1:
    #     if_not_iterable_convert_2_type_variable = list_types_iterable_allowed[0]
    # else:
    #     if if_not_iterable_convert_2_type_variable not in list_types_iterable_allowed:
    #         raise Exception(
    #             'if_not_iterable_convert_2_type_variable is not in list_types_iterable_allowed.\n'
    #             'Now, if_not_iterable_convert_2_type_variable = {} and list_types_iterable_allowed = {}.\n'
    #             'By default, make_sure_is_iterable(*variables, list_types_iterable_allowed=[list, tuple, np.ndarray],\n'
    #             'if_not_iterable_convert_2_type_variable=np.ndarray)'.format(
    #                 if_not_iterable_convert_2_type_variable, list_types_iterable_allowed))

    variables = list(variables)
    n_variables = len(variables)

    for i in range(n_variables):

        try:
            len(variables[i])

        except TypeError:

            if if_not_iterable_convert_2_type_variable == str:
                variables[i] = str(variables[i])

            else:
                variables[i] = [variables[i]]

                if if_not_iterable_convert_2_type_variable == list:
                    variables[i] = list(variables[i])

                elif if_not_iterable_convert_2_type_variable == tuple:
                    variables[i] = tuple(variables[i])

                elif if_not_iterable_convert_2_type_variable == np.ndarray:
                    variables[i] = np.array(variables[i])
                else:
                    raise Exception(
                        'if_not_iterable_convert_2_type_variable has to either be equal to np.ndarray, list, tuple or str.\n'
                        'Now, if_not_iterable_convert_2_type_variable = {}.\n'
                        'By default, if_not_iterable_convert_2_type_variable=np.ndarray'.format(
                            if_not_iterable_convert_2_type_variable))

        # type_i_variable = type(variables[i])
        #
        # # if type_i_variable not in list_types_iterable_allowed:
        #     if if_not_iterable_convert_2_type_variable == list:
        #         variables[i] = list([variables[i]])
        #
        #     elif if_not_iterable_convert_2_type_variable == tuple:
        #         variables[i] = tuple([variables[i]])
        #
        #     elif if_not_iterable_convert_2_type_variable == np.ndarray:
        #         variables[i] = np.array([variables[i]])
        #     else:
        #         raise Exception(
        #             'if_not_iterable_convert_2_type_variable has to either be equal to np.ndarray, list or tuple.\n'
        #             'Now, if_not_iterable_convert_2_type_variable = {}.\n'
        #             'By default, if_not_iterable_convert_2_type_variable=np.ndarray'.format(
        #                 if_not_iterable_convert_2_type_variable))

        # elif type_i_variable == np.ndarray:
        #     if variables[i].ndim == 0:
        #         variables[i] = np.array([variables[i]])

    if n_variables == 1:
        return variables[0]
    else:
        return variables


def transfer_n_random_samples_from_arr1_to_arr2(arr1, arr2, n_samples=1, axis=0, replace=False):
    # def transfer_n_random_samples_from_arr1_to_arr2(arr1, arr2, first_n_samples=1, all=False):

    # SPLIT THIS IN TWO FUNCTIONS:
    # 1) transfer_n_random_samples_from_arr1_to_arr2;
    # 2) transfer_n_samples_from_arr1_to_arr2;

    arr1, arr2 = make_sure_is_iterable(arr1, arr2)


    shape_arr1 = np.array(arr1.shape)
    shape_arr2 = np.array(arr2.shape)

    n_dimensions_arr1 = len(shape_arr1)
    n_dimensions_arr2 = len(shape_arr2)
    if n_dimensions_arr1 != n_dimensions_arr2:

        if n_dimensions_arr1 == n_dimensions_arr2 - 1:

            arr1 = np.expand_dims(arr1, axis=axis)
            shape_arr1 = np.array(arr1.shape)
            n_dimensions_arr1 = len(shape_arr1)

        else:
            raise Exception('n_dimensions_arr1 has to be equal to n_dimensions_arr2\n'
                            'or to n_dimensions_arr2 - 1.\n'
                            'Now, n_dimensions_arr1 = {} and n_dimensions_arr2 = {}.'.format(
                n_dimensions_arr1, n_dimensions_arr2))

    dimensions_arr1_and_arr2 = np.arange(n_dimensions_arr1)

    if np.all(axis != dimensions_arr1_and_arr2):
        raise Exception('axis is not in the dimensions_arr1_and_arr2.\n'
                        'axis has to be equal to either one ot the\n'
                        'dimensions_arr1_and_arr2.\n'
                        'Now, axis = {} and dimensions_arr1_and_arr2 = {}.\n'
                        'By default, axis=0.'.format(
            axis, dimensions_arr1_and_arr2))

    n_samples_arr1 = shape_arr1[axis]
    n_samples_arr2 = shape_arr2[axis]
    if n_samples > n_samples_arr1 and replace is False:
        raise Exception('n_samples cannot be greater than n_samples of arr1, when replace=False.\n'
                        'Now, n_samples = {}, n_samples_arr1 = {} and replace = {}.'.format(
            n_samples, n_samples_arr1, replace))


    dimensions_within_samples = np.argwhere(dimensions_arr1_and_arr2 != axis)[:, 0]
    # dimensions_within_samples = dimensions_within_samples.astype(int)

    shape_1_sample_arr1 = shape_arr1[dimensions_within_samples]
    shape_1_sample_arr2 = shape_arr2[dimensions_within_samples]
    if np.any(shape_1_sample_arr1 != shape_1_sample_arr2):

        raise Exception('shape_1_sample_arr1 has to be equal to shape_1_sample_arr2.\n'
                        'Now,shape_1_sample_arr1 = {} and shape_1_samples_arr2 = {}.'.format(
            shape_1_sample_arr1, shape_1_sample_arr2))

    indexes_chosen_samples_in_arr1 = np.random.choice(n_samples_arr1, n_samples, replace=replace)

    indexes = np.empty(n_dimensions_arr1, dtype=object)
    for i_d in range(n_dimensions_arr1):
        if i_d == axis:
            # indexes_samples_i_window.append(slice(0, n_samples_window))
            indexes[i_d] = indexes_chosen_samples_in_arr1
        else:
            indexes[i_d] = np.arange(shape_arr1[i_d])

    # if n_dimensions_arr1 == 1:
    #     chosen_sample_in_arr1 = arr1[np.ix_(indexes)]
    # else:
    chosen_sample_in_arr1 = arr1[np.ix_(*indexes)]

    arr2 = np.append(arr2, chosen_sample_in_arr1)

    arr1 = np.delete(arr1, indexes_chosen_samples_in_arr1, axis=axis)
    # indexes_chosen_sample_in_arr1 = np.argwhere(arr1 == chosen_sample_in_arr1)[0]

    # n_chosen_samples_in_arr1 = len(indexes_chosen_sample_in_arr1)
    # if all:
    #
    # elif n_chosen_samples_in_arr1 <= first_n_samples:
    #     indexes_chosen_sample_in_arr1 = indexes_chosen_sample_in_arr1[np.arange(n_chosen_samples_in_arr1)]
    #
    # elif n_chosen_samples_in_arr1 > first_n_samples:
    #     indexes_chosen_sample_in_arr1 = indexes_chosen_sample_in_arr1[np.arange(first_n_samples)]

    # arr1 = np.delete(arr1, indexes_chosen_sample_in_arr1)

    return arr1, arr2


def shuffle_in_windows(data, n_samples_window=None, n_windows=None, axis=0, dtype=None):

    type_data = type(data)
    if type_data != np.ndarray:
        raise Exception('data must be an numpy array. '
                        'Now, the class of data is {}'.format(type_data))

    shape_data = data.shape
    n_samples_data = shape_data[axis]
    n_d_data = len(shape_data)
    if axis > n_d_data:
        raise Exception(
            'shuffle in dimension {} can\'t be done,\n'
            'because data has {} dimensions and\n'
            'it doesn\'t have the {}th dimension.\n'
            'axis has to be an integer in the interval\n'
            '0 =< axis < n_d_data.\n'
            'In this case, axis = {} and n_d_data = {}'.format(
                axis, n_d_data, axis, axis, n_d_data))

    if n_samples_window is None and n_windows is None:
        raise Exception('Both n_samples_window and n_windows are None.'
                        'Define only one of them with an integer.'
                        'One of them has to be None and the other one has to be an integer')

    elif n_samples_window is None and n_windows is not None:
        n_samples_window = math.ceil(n_samples_data / n_windows)

    elif n_samples_window is not None and n_windows is None:
        n_windows = math.ceil(n_samples_data / n_samples_window)

    elif n_samples_window is not None and n_windows is not None:

        tmp_n_samples_window = math.ceil(n_samples_data / n_windows)
        tmp_n_windows = math.ceil(n_samples_data / n_samples_window)

        if tmp_n_samples_window != n_samples_window and tmp_n_windows != n_windows:

            raise Exception(
                'For n_samples_data = {}, n_samples_window = {} and n_windows = {} are not compatible.\n'
                'If n_samples_window = {}, n_windows has to be {} or None.\n'
                'If n_windows = {}, n_samples_window has to be {} or None'.format(
                    n_samples_data, n_samples_window, n_windows, n_samples_window,
                    tmp_n_windows, n_windows, tmp_n_samples_window))

    if n_samples_window * n_windows > n_samples_data:
        print('Warning: n_samples_window * n_windows > n_samples_data.\n'
              'n_samples_window * n_windows = {} and n_samples_data = {}'.format(
            n_samples_window * n_windows, n_samples_data))

    indexes_i_window = np.empty(n_d_data, dtype=object)
    for i_d in range(n_d_data):
        if i_d != axis:
            indexes_i_window[i_d] = slice(0, shape_data[i_d], 1)

    if dtype is None:
        dtype = data.dtype
    data_shuffle = np.empty(shape_data, dtype=dtype)
    start = 0
    end = n_samples_window
    for i_window in range(n_windows):

        indexes_i_window[axis] = slice(start, end, 1)
        indexes_tuple = tuple(indexes_i_window)
        data_shuffle[indexes_tuple] = shuffle_in_any_dimension(data[indexes_tuple], axis=axis)
        start = end
        end += n_samples_window

    return data_shuffle


def shuffle_in_any_dimension(data, axis=0):

    type_data = type(data)
    if type_data != np.ndarray:
        raise Exception('data must be an numpy array. '
                        'Now, the class of data is {}'.format(type_data))

    shape_data = data.shape
    n_samples = shape_data[axis]
    n_axes = len(shape_data)

    if axis > n_axes:
        raise Exception(
            'shuffle in axis {} can\'t be done,\n'
            'because data has {} axes and\n'
            'it doesn\'t have the {}th axis.\n'
            'axis has to be an integer in the interval\n'
            '0 =< axis < n_axes.\n'
            'In this case, axis = {} and n_axes = {}'.format(
                axis, n_axes, axis, axis, n_axes))


    indexes_new = np.empty(n_axes, dtype=object)
    for a in range(n_axes):
        if a == axis:
            # indexes_samples_i_window.append(slice(0, n_samples_window))
            indexes_new[a] = np.random.rand(n_samples).argsort()
        else:
            indexes_new[a] = slice(0, shape_data[a], 1)

    data_shuffle = data[tuple(indexes_new)]

    return data_shuffle


def split_by_n(array, n_slices, axis=0):

    shape_array = np.array(array.shape)

    n_dimensions_array = len(shape_array)

    indexes = np.empty(n_dimensions_array, dtype=object)

    for i_d in range(n_dimensions_array):
        if i_d != axis:

            indexes[i_d] = slice(None)

    # indexes_samples_i_window.append(slice(0, n_samples_window))

    n_elements_array = shape_array[axis]

    n_elements_slice = n_elements_array / n_slices

    sliced_array = np.empty(n_slices, dtype=object)

    index_start = 0
    for i_slice in range(n_slices):

        index_stop = int((i_slice + 1) * n_elements_slice)

        indexes[axis] = slice(index_start, index_stop)

        # sliced_array[i_slice] = array[int(i_slice * n_elements_slice): int((i_slice + 1) * n_elements_slice)]
        sliced_array[i_slice] = array[tuple(indexes)]

        index_start = index_stop

    return sliced_array


def split_by_percentages(array, p_samples, axis=0, shuffle=False, sort=False):

    N = array.shape[axis]
    S = len(p_samples)
    M = np.empty(S, dtype='i')

    for s in range(S):
        M[s] = cp_maths.rint(N / 100 * p_samples[s])

    sum_M = M.sum()
    if sum_M != N:
        if sum_M < N:
            s = 0
            M[s] += 1
            while sum_M < N:
                s += 1
                M[s] += 1
        else:
            s = 0
            M[s] -= 1
            while sum_M < N:
                s += 1
                M[s] -= 1


    indexes_splitting = np.empty(S - 1, dtype='i')
    for s in range(S - 1):
        indexes_splitting[s] = M[slice(0, s + 1, 1)].sum()

    if shuffle:
        n_axes = len(array.shape)
        indexes_array = np.empty(n_axes, dtype=object)
        indexes_array[:] = slice(0, None, 1)
        indexes_array[axis] = np.random.permutation(N)
        arrays = np.split(array[tuple(indexes_array)], indexes_splitting, axis=axis)

        # arrays = np.split(shuffle_in_any_dimension(array, axis=axis), indexes_splitting, axis=axis)
        if sort:
            indexes_perm = np.split(indexes_array[axis], indexes_splitting, axis=axis)
            indexes_recover = [None] * S  # type: list
            # arrays_new = [None] * S  # type: list
            for s in range(S):
                indexes_recover[s] = np.empty(M[s], dtype='i')
                start_s = 0
                for m in range(M[s]):
                    for i in range(start_s, N):
                        indexes_found = np.where(i == indexes_perm[s])[0]
                        n_indexes_found = len(indexes_found)
                        if n_indexes_found == 1:
                            indexes_recover[s][m] = indexes_found[0]
                            # np.delete(indexes_perm[s], indexes_found)
                            start_s = i + 1
                            break

                indexes_array[axis] = indexes_recover[s]
                arrays[s] = arrays[s][tuple(indexes_array)]
                # arrays_new[s] = arrays[s][tuple(indexes_array)]
                # indexes_recover[s] = [np.where(i == indexes_perm[s])[0][0] for i in range(N)]
                # c = b[indexes_recover]
                # arrays[s].sort()
    else:
        arrays = np.split(array, indexes_splitting, axis=axis)

    return arrays


def pad_array_from_n_samples_target(array, n_samples_target=1, axis=0):

    try:
        len(array)
        array = np.asarray(array)
    except TypeError:
        array = np.asarray([array])

    shape_array = np.asarray(array.shape)
    n_axes = len(shape_array)
    while n_axes <= axis:
        array = np.expand_dims(array, axis=n_axes)
        shape_array = np.asarray(array.shape)
        n_axes = len(shape_array)

    n_samples_true = shape_array[axis]
    while n_samples_true < n_samples_target:
        array = np.append(array, array, axis=axis)
        n_samples_true = array.shape[axis]

    index_array = np.full(n_axes, slice(None))
    index_array[axis] = slice(n_samples_target)
    array = array[tuple(index_array)]

    return array


def pad_arr_1_from_arr_2(arr_1, arr_2, axis=0):

    try:
        len(arr_1)
        arr_1 = np.asarray(arr_1)
    except TypeError:
        arr_1 = np.asarray([arr_1])

    try:
        len(arr_2)
        arr_2 = np.asarray(arr_2)
    except TypeError:
        arr_2 = np.asarray([arr_2])

    shape_arr_1 = np.asarray(arr_1.shape)
    n_axes_1 = len(shape_arr_1)
    shape_arr_2 = np.asarray(arr_2.shape)
    n_axes_2 = len(shape_arr_2)

    if n_axes_1 == n_axes_2 - 1:
        arr_1 = np.expand_dims(arr_1, axis=axis)
        shape_arr_1 = np.asarray(arr_1.shape)
        n_axes_1 = len(shape_arr_1)
    elif n_axes_1 == n_axes_2:
        pass
    else:
        raise Exception('dimensions of arr_1 and arr_2 do not match')

    if np.all(shape_arr_1[np.arange(n_axes_1) != axis] == shape_arr_2[np.arange(n_axes_2) != axis]):
        pass
    else:
        raise Exception('dimensions of arr_1 and arr_2 do not match')

    n_samples_true = shape_arr_1[axis]
    n_samples_target = shape_arr_2[axis]

    while n_samples_true < n_samples_target:
        arr_1 = np.append(arr_1, arr_1, axis=axis)
        n_samples_true = arr_1.shape[axis]

    index_arr_1 = np.full(n_axes_1, slice(None))
    index_arr_1[axis] = slice(n_samples_target)

    arr_1 = arr_1[tuple(index_arr_1)]

    return arr_1


def centered_indexes_to_normal_indexes(
        centered_indexes, axis_indexes, shape_array):

    axis_dimensions_of_array = int(not (axis_indexes == 1))

    if isinstance(centered_indexes, list) or isinstance(centered_indexes, tuple):
        centered_indexes = np.asarray(centered_indexes)
        n_axes_indexes = len(centered_indexes.shape)
    elif isinstance(centered_indexes, np.ndarray):
        n_axes_indexes = len(centered_indexes.shape)
    else:
        n_axes_indexes = 0

    if isinstance(shape_array, list) or isinstance(shape_array, tuple):
        shape_array = np.asarray(shape_array)
        n_axes_shape_array = shape_array.size
    elif isinstance(shape_array, np.ndarray):
        n_axes_shape_array = shape_array.size
    else:
        n_axes_shape_array = 0

    if (n_axes_indexes == 2) and (n_axes_shape_array == 1):
        shape_array = np.expand_dims(shape_array, axis=axis_indexes)

    normal_indexes = (shape_array / 2) + centered_indexes
    if isinstance(normal_indexes, np.ndarray):
        normal_indexes = np.floor(normal_indexes)
    else:
        normal_indexes = math.floor(normal_indexes)

    return normal_indexes


class IntRange:

    def __init__(self, range_raw):

        if isinstance(range_raw, IntRange):
            start = range_raw.start
            stop = range_raw.stop
            step = range_raw.step

        elif isinstance(range_raw, dict):
            keys = ['start', 'stop', 'step']
            for k in range_raw.keys():
                if k in keys:
                    continue
                else:
                    raise KeyError(k)
            if range_raw.get('start') is None:
                start = 0
            else:
                start = range_raw['start']

            if range_raw.get('stop') is None:
                raise ValueError("range_raw['stop']")
            else:
                stop = range_raw['stop']

            if range_raw.get('step') is None:
                step = 1
            else:
                step = range_raw['step']

        elif isinstance(range_raw, range):
            start = range_raw.start
            stop = range_raw.stop
            step = range_raw.step
        elif isinstance(range_raw, (list, tuple, np.ndarray)):
            n_elements = len(range_raw)
            if n_elements == 1:
                start = 0
                stop = range_raw[0]
                step = 1
            elif n_elements == 2:
                start, stop = range_raw
                step = 1
            elif n_elements == 3:
                start, stop, step = range_raw
            else:
                raise ValueError('range_raw has to have at least 1 element. {} are given'.format(n_elements))
        elif isinstance(range_raw, numbers.Number):
            start = 0
            stop = range_raw
            step = 1
        else:
            raise TypeError('range_raw')

        self.start = start
        self.stop = stop
        self.step = step
        if self.stop > self.start:
            if self.step > 0:
                self.len = math.ceil((self.stop - self.start) / self.step)
            elif self.step < 0:
                self.len = 0
            else:
                raise ValueError('self.step')
        elif self.stop < self.start:
            if self.step < 0:
                self.len = math.ceil(abs((self.stop - self.start) / self.step))
            elif self.step > 0:
                self.len = 0
            else:
                raise ValueError('self.step')
        else:
            self.len = 0

    def to_slice(self):
        return slice(self.start, self.stop, self.step)

    def to_list(self):
        return list(range(self.start, self.stop, self.step))


class FloatRange:
    def __init__(self, range_raw):
        if isinstance(range_raw, dict):
            keys = ['start', 'stop', 'step']
            for k in range_raw.keys():
                if k in keys:
                    continue
                else:
                    raise KeyError(k)
            if range_raw.get('start') is None:
                start = 0
            else:
                start = range_raw['start']

            if range_raw.get('stop') is None:
                raise ValueError("range_raw['stop']")
            else:
                stop = range_raw['stop']

            if range_raw.get('step') is None:
                step = 1
            else:
                step = range_raw['step']

        elif isinstance(range_raw, range):
            start = range_raw.start
            stop = range_raw.stop
            step = range_raw.step
        elif isinstance(range_raw, (list, tuple, np.ndarray)):
            n_elements = len(range_raw)
            if n_elements == 1:
                start = 0
                stop = range_raw[0]
                step = 1
            elif n_elements == 2:
                start, stop = range_raw
                step = 1
            elif n_elements == 3:
                start, stop, step = range_raw
            else:
                raise ValueError('range_raw has to have at least 1 element. {} are given'.format(n_elements))
        elif isinstance(range_raw, numbers.Number) or range_raw is None:
            start = 0
            stop = range_raw
            step = 1
        else:
            raise TypeError('range_raw')

        self.start = cp_maths.convert_to_int_or_float(start)
        self.stop = cp_maths.convert_to_int_or_float(stop)
        self.step = cp_maths.convert_to_int_or_float(step)
        self.len = abs(cp_maths.convert_to_int_or_float((self.stop - self.start) / self.step))

    # def __init__(self, stop, *args):
    #
    #     n_args = len(args)
    #     if n_args == 0:
    #         self.start = 0
    #         self.stop = stop
    #         self.step = 1
    #     elif n_args == 1:
    #         self.start = stop
    #         self.stop = args[0]
    #         self.step = 1
    #     elif n_args == 2:
    #         self.start = stop
    #         self.stop = args[0]
    #         self.step = args[1]
    #     else:
    #         raise TypeError('MyRange expected at most 3 arguments, got {}'.format(n_args + 1))
    #
    #     self.len = cp_maths.convert_to_int_or_float((self.stop - self.start) / self.step)
    #     # self.len = (self.stop - self.start) / self.step

    def starts_of_n_within_equal_ranges(self, n):

        len_each_equal_ranges = self.len / n
        starts = np.arange(
            start=self.start,
            stop=self.stop - (len_each_equal_ranges * .1),
            step=len_each_equal_ranges)
        return starts

    def stops_of_n_within_equal_ranges(self, n):

        len_each_equal_ranges = self.len / n
        stops = np.arange(
            start=self.start + len_each_equal_ranges,
            stop=self.stop + len_each_equal_ranges - (len_each_equal_ranges * .1),
            step=len_each_equal_ranges)

        return stops

    def centers_of_n_within_equal_ranges(self, n):

        len_each_equal_ranges = self.len / n
        half_len_each_equal_ranges = len_each_equal_ranges / 2
        centers = np.arange(
            start=self.start + half_len_each_equal_ranges,
            stop=self.stop + half_len_each_equal_ranges - (len_each_equal_ranges * .1),
            step=len_each_equal_ranges)
        return centers


def select_dtype_from_2_arrays(x, y):

    if isinstance(x, np.ndarray):
        dtype_x = x.dtype
    else:
        dtype_x = np.asarray(x).dtype

    if isinstance(y, np.ndarray):
        dtype_y = y.dtype
    else:
        dtype_y = np.asarray(y).dtype

    if dtype_x.kind == dtype_y.kind:
        if int(dtype_x.str[2:]) > int(dtype_y.str[2:]):
            dtype_out = dtype_x.str
        else:
            dtype_out = dtype_y.str

    elif (dtype_x.kind == 'f') and (dtype_y.kind == 'i'):
        dtype_out = dtype_x.str
    elif (dtype_x.kind == 'i') and (dtype_y.kind == 'f'):
        dtype_out = dtype_y.str
    else:
        dtype_out = 'O'

    return dtype_out
