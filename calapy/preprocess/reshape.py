import numpy as np
from ..combinations import n_conditions_to_combinations, conditions_to_combinations
from ..array import samples_in_arr1_are_in_arr2
from ..format import seq_of_numeric_indexes_to_seq_of_slices


def merge_axes(
        array_input,
        axes_removing_input,
        axis_pooling_input,
        dtype=None):
    # Input requirements:
    # 1) axes_removing_input cannot contain same values;
    # 2) axes_removing_input[a] != axis_pooling_input

    # Import requirements:
    # 1) import numpy as np
    # 2) from ccalafiore.combinations import n_conditions_to_combinations
    # 3) from ccalafiore.array import samples_in_arr1_are_in_arr2

    # format axes_removing_input
    if isinstance(axes_removing_input, list) or isinstance(axes_removing_input, tuple):
        axes_removing_input = np.asarray(axes_removing_input, dtype=int)
        n_axes_removing_input = axes_removing_input.size
        if n_axes_removing_input == 1:
            axes_removing_input = axes_removing_input[0]
    elif isinstance(axes_removing_input, np.ndarray):
        if len(axes_removing_input.shape) == 0:
            n_axes_removing_input = 1
        else:
            n_axes_removing_input = axes_removing_input.size
            if n_axes_removing_input == 1:
                axes_removing_input = axes_removing_input[0]
    else:
        n_axes_removing_input = 1

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = shape_array_input.size
    if axis_pooling_input < 0:
        axis_pooling_input += n_axes_array_input
    axis_pooling_output = axis_pooling_input - np.sum(axes_removing_input < axis_pooling_input)

    # check point 1
    if n_axes_removing_input > 1:
        for a in axes_removing_input:
            if np.sum(a == axes_removing_input) > 1:
                raise ValueError('axes_removing_input cannot contain repeated values. {} is repeated.'.format(a))

    # check point 2
    if np.sum(axes_removing_input == axis_pooling_input) > 0:
        raise ValueError('The following condition is not met:\n'
                         '\taxes_removing_input[a] \u2260 axis_pooling_input')

    if n_axes_removing_input == 1:
        if axes_removing_input < 0:
            axes_removing_input += n_axes_array_input
        if axes_removing_input < 0 or axes_removing_input >= n_axes_array_input:
            raise np.AxisError(
                'axes_removing_input = {} is out of range for array_input of {} dimensions'.format(
                    axes_removing_input, n_axes_array_input))
        n_conditions_in_axes_removing_input = [shape_array_input[axes_removing_input]]
    elif n_axes_removing_input > 1:
        axes_removing_input[axes_removing_input < 0] += n_axes_array_input
        if np.any(axes_removing_input < 0) or np.any(axes_removing_input >= n_axes_array_input):
            raise np.AxisError(
                'axes_removing_input = {} is out of range for array_input of {} dimensions'.format(
                    axes_removing_input, n_axes_array_input))
        n_conditions_in_axes_removing_input = shape_array_input[axes_removing_input]
    
    axes_merging_input = np.append(axis_pooling_input, axes_removing_input)
    # n_axes_merging_input = axes_merging_input.size

    axes_array_input = np.arange(n_axes_array_input)
    axes_other_array_input = axes_array_input[np.logical_not(samples_in_arr1_are_in_arr2(
        axes_array_input, axes_merging_input))]

    n_axes_array_output = n_axes_array_input - n_axes_removing_input
    axes_array_output = np.arange(n_axes_array_output)
    axes_other_output = axes_array_output[axes_array_output != axis_pooling_output]

    shape_array_output = np.empty(n_axes_array_output, dtype=int)
    n_conditions_in_axes_merging_input = shape_array_input[axes_merging_input]
    shape_array_output[axis_pooling_output] = np.prod(n_conditions_in_axes_merging_input)
    shape_array_output[axes_other_output] = shape_array_input[axes_other_array_input]
    if dtype is None:
        dtype = array_input.dtype
    array_output = np.empty(shape_array_output, dtype=dtype)

    axis_combinations = 0
    axis_variables = 1
    combinations_axes_removing_input = n_conditions_to_combinations(
        n_conditions_in_axes_removing_input, axis_combinations=axis_combinations, order_variables='lr')
    n_combinations_axes_removing_input = combinations_axes_removing_input.shape[axis_combinations]

    indexes_combinations = np.empty(2, dtype=object)
    if n_axes_removing_input == 1:
        indexes_combinations[axis_variables] = 0
    elif n_axes_removing_input > 1:
        indexes_combinations[axis_variables] = slice(0, n_axes_removing_input, 1)

    indexes_input = np.empty(n_axes_array_input, dtype=object)
    indexes_input[:] = slice(0, None, 1)

    indexes_output = np.empty(n_axes_array_output, dtype=object)
    indexes_output[:] = slice(0, None, 1)

    length_axis_pooling_input = shape_array_input[axis_pooling_input]
    start_index_axis_pooling_output = 0
    for c in range(n_combinations_axes_removing_input):
        stop_index_axis_pooling_output = (c + 1) * length_axis_pooling_input
        indexes_output[axis_pooling_output] = slice(
            start_index_axis_pooling_output, stop_index_axis_pooling_output)
        start_index_axis_pooling_output = stop_index_axis_pooling_output

        indexes_combinations[axis_combinations] = c
        indexes_input[axes_removing_input] = (
            combinations_axes_removing_input[tuple(indexes_combinations)])
        array_output[tuple(indexes_output)] = array_input[tuple(indexes_input)]

    return array_output


def merge_axes_and_add_axes_removing_info_to_axis_receiving(
        array_input,
        axes_removing_input,
        axis_pooling_input,
        axis_receiving_input,
        indexes_inserting_output=None,
        dtype=None):

    # Input requirements:
    # 1) axes_removing_input cannot contain same values;
    # 2) indexes_inserting_output cannot contain same values;
    # 3) if indexes_inserting_output is not None,
    #        shapes of axes_removing_input and indexes_inserting_output must be equal
    # 4) axes_removing_input[a] != axis_pooling_input != axis_receiving_input

    # Import requirements:
    # 1) import numpy as np
    # 2) from ccalafiore.combinations import n_conditions_to_combinations
    # 3) from ccalafiore.array import samples_in_arr1_are_in_arr2

    # format axes_removing_input
    if isinstance(axes_removing_input, list) or isinstance(axes_removing_input, tuple):
        axes_removing_input = np.asarray(axes_removing_input, dtype=int)
        n_axes_removing_input = axes_removing_input.size
        if n_axes_removing_input == 1:
            axes_removing_input = axes_removing_input[0]
    elif isinstance(axes_removing_input, np.ndarray):
        if len(axes_removing_input.shape) == 0:
            n_axes_removing_input = 1
        else:
            n_axes_removing_input = axes_removing_input.size
            if n_axes_removing_input == 1:
                axes_removing_input = axes_removing_input[0]
    else:
        n_axes_removing_input = 1

    # format indexes_inserting_output
    if indexes_inserting_output is None:
        n_indexes_inserting_output = n_axes_removing_input
        if n_indexes_inserting_output == 1:
            indexes_inserting_output = 0
        else:
            indexes_inserting_output = np.arange(0, n_indexes_inserting_output, 1, dtype=int)
    elif isinstance(indexes_inserting_output, list) or isinstance(indexes_inserting_output, tuple):
        indexes_inserting_output = np.asarray(indexes_inserting_output, dtype=int)
        n_indexes_inserting_output = indexes_inserting_output.size
        if n_indexes_inserting_output == 1:
            indexes_inserting_output = indexes_inserting_output[0]
    elif isinstance(indexes_inserting_output, np.ndarray):
        if len(indexes_inserting_output.shape) == 0:
            n_indexes_inserting_output = 1
        else:
            n_indexes_inserting_output = indexes_inserting_output.size
            if n_indexes_inserting_output == 1:
                indexes_inserting_output = indexes_inserting_output[0]
    else:
        n_indexes_inserting_output = 1

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = shape_array_input.size
    if axis_pooling_input < 0:
        axis_pooling_input += n_axes_array_input
    if axis_receiving_input < 0:
        axis_receiving_input += n_axes_array_input
    length_axis_receiving_input = shape_array_input[axis_receiving_input]
    # indexes_in_axis_receiving_input = np.arange(length_axis_receiving_input)
    length_axis_receiving_output = length_axis_receiving_input + n_indexes_inserting_output
    axis_receiving_output = axis_receiving_input - np.sum(axes_removing_input < axis_receiving_input)
    axis_pooling_output = axis_pooling_input - np.sum(axes_removing_input < axis_pooling_input)

    # check point 1
    if n_axes_removing_input > 1:
        for a in axes_removing_input:
            if np.sum(a == axes_removing_input) > 1:
                raise ValueError('axes_removing_input cannot contain repeated values. {} is repeated.'.format(a))

    # check point 2
    if n_indexes_inserting_output > 1:
        for i in indexes_inserting_output:
            if np.sum(i == indexes_inserting_output) > 1:
                raise ValueError('indexes_inserting_output cannot contain repeated values. {} is repeated.'.format(i))

    # check point 3
    if n_indexes_inserting_output != n_axes_removing_input:
        raise ValueError(
            'Shapes of axes_removing_input and indexes_inserting_output must be equal')

    # check point 4
    if np.sum(axes_removing_input == axis_pooling_input) > 0:
        raise ValueError('The following condition is not met:\n'
                         '\taxes_removing_input[a] \u2260 axis_pooling_input')
    if np.sum(axes_removing_input == axis_receiving_input) > 0:
        raise ValueError('The following condition is not met:\n'
                         '\taxes_removing_input[a] \u2260 axis_receiving_input')
    if axis_pooling_input == axis_receiving_input:
        raise ValueError('The following condition is not met:\n'
                         '\taxis_pooling_input \u2260 axis_receiving_input')

    if n_axes_removing_input == 1:
        if axes_removing_input < 0:
            axes_removing_input += n_axes_array_input
        if axes_removing_input < 0 or axes_removing_input >= n_axes_array_input:
            raise np.AxisError(
                'axes_removing_input = {} is out of range for array_input of {} dimensions'.format(
                    axes_removing_input, n_axes_array_input))
        n_conditions_in_axes_removing_input = [shape_array_input[axes_removing_input]]

        if indexes_inserting_output < 0:
            indexes_inserting_output += length_axis_receiving_output
        if indexes_inserting_output < 0 or indexes_inserting_output >= length_axis_receiving_output:
            raise IndexError(
                'indexes_inserting_output = {} is out of range for axis_receiving_output = {} with size = {}'.format(
                    indexes_inserting_output, axis_receiving_output, length_axis_receiving_output))
    elif n_axes_removing_input > 1:
        axes_removing_input[axes_removing_input < 0] += n_axes_array_input
        if np.any(axes_removing_input < 0) or np.any(axes_removing_input >= n_axes_array_input):
            raise np.AxisError(
                'axes_removing_input = {} is out of range for array_input of {} dimensions'.format(
                    axes_removing_input, n_axes_array_input))
        n_conditions_in_axes_removing_input = shape_array_input[axes_removing_input]

        indexes_inserting_output[indexes_inserting_output < 0] += length_axis_receiving_output
        if np.any(indexes_inserting_output < 0) or np.any(indexes_inserting_output >= length_axis_receiving_output):
            raise IndexError(
                'indexes_inserting_output = {} is out of range for axis_receiving_output = {} with size = {}'.format(
                    indexes_inserting_output, axis_receiving_output, length_axis_receiving_output))

    axes_merging_input = np.append(axis_pooling_input, axes_removing_input)
    # n_axes_merging_input = axes_merging_input.size

    axes_array_input = np.arange(n_axes_array_input)
    axes_other_array_input = axes_array_input[np.logical_not(samples_in_arr1_are_in_arr2(
        axes_array_input, np.append(axes_merging_input, axis_receiving_input)))]

    n_axes_array_output = n_axes_array_input - n_axes_removing_input
    axes_array_output = np.arange(n_axes_array_output)
    axes_other_output = axes_array_output[np.logical_not(samples_in_arr1_are_in_arr2(
        axes_array_output, [axis_pooling_output, axis_receiving_output]))]
    shape_array_output = np.empty(n_axes_array_output, dtype=int)
    n_conditions_in_axes_merging_input = shape_array_input[axes_merging_input]
    shape_array_output[axis_pooling_output] = np.prod(n_conditions_in_axes_merging_input)
    shape_array_output[axis_receiving_output] = length_axis_receiving_output
    shape_array_output[axes_other_output] = shape_array_input[axes_other_array_input]
    if dtype is None:
        dtype = array_input.dtype
    array_output = np.empty(shape_array_output, dtype=dtype)

    indexes_in_axis_receiving_output = np.arange(length_axis_receiving_output)
    indexes_in_axis_receiving_output_from_axis_receiving_input = indexes_in_axis_receiving_output[np.logical_not(
        samples_in_arr1_are_in_arr2(indexes_in_axis_receiving_output, indexes_inserting_output))]

    # axis_variables = int(axis_receiving_input > axis_pooling_input)
    # axis_combinations = int(axis_variables == 0)
    axis_combinations = 0
    axis_variables = 1
    combinations_axes_removing_input = n_conditions_to_combinations(
        n_conditions_in_axes_removing_input, axis_combinations=axis_combinations, order_variables='lr')
    n_combinations_axes_removing_input = combinations_axes_removing_input.shape[axis_combinations]

    indexes_combinations = np.empty(2, dtype=object)

    indexes_input = np.empty(n_axes_array_input, dtype=object)
    indexes_input[:] = slice(0, None, 1)
    if length_axis_receiving_input == 1:
        indexes_input[axis_receiving_input] = 0

    indexes_output1 = np.empty(n_axes_array_output, dtype=object)
    indexes_output1[:] = slice(0, None, 1)
    indexes_output2 = np.copy(indexes_output1)
    indexes_output1[axis_receiving_output], indexes_output2[axis_receiving_output] = (
        seq_of_numeric_indexes_to_seq_of_slices(
            [indexes_in_axis_receiving_output_from_axis_receiving_input, indexes_inserting_output]))

    length_axis_pooling_input = shape_array_input[axis_pooling_input]
    start_index_axis_pooling_output = 0

    if n_axes_removing_input > 1:

        indexes_combinations[axis_variables] = slice(0, n_axes_removing_input, 1)

        shape_combinations_axes_removing_input_c = np.empty(n_axes_array_output, dtype=int)
        shape_combinations_axes_removing_input_c[:] = 1
        shape_combinations_axes_removing_input_c[axis_receiving_output] = n_indexes_inserting_output
        combinations_axes_removing_input_c = np.empty(shape_combinations_axes_removing_input_c, dtype=int)
        indexes_combinations_axes_removing_input_c = np.empty(n_axes_array_output, dtype=object)
        indexes_combinations_axes_removing_input_c[:] = 0
        indexes_combinations_axes_removing_input_c[axis_receiving_output] = slice(0, n_indexes_inserting_output, 1)
        indexes_combinations_axes_removing_input_c_tuple = tuple(indexes_combinations_axes_removing_input_c)

        for c in range(n_combinations_axes_removing_input):
            stop_index_axis_pooling_output = (c + 1) * length_axis_pooling_input
            indexes_output1[axis_pooling_output] = indexes_output2[axis_pooling_output] = slice(
                start_index_axis_pooling_output, stop_index_axis_pooling_output)
            start_index_axis_pooling_output = stop_index_axis_pooling_output

            indexes_combinations[axis_combinations] = c
            indexes_input[axes_removing_input] = (
                combinations_axes_removing_input[tuple(indexes_combinations)])
            array_output[tuple(indexes_output1)] = array_input[tuple(indexes_input)]

            combinations_axes_removing_input_c[indexes_combinations_axes_removing_input_c_tuple] = (
                combinations_axes_removing_input[tuple(indexes_combinations)])

            array_output[tuple(indexes_output2)] = combinations_axes_removing_input_c

    elif n_axes_removing_input == 1:

        indexes_combinations[axis_variables] = 0

        for c in range(n_combinations_axes_removing_input):
            stop_index_axis_pooling_output = (c + 1) * length_axis_pooling_input
            indexes_output1[axis_pooling_output] = indexes_output2[axis_pooling_output] = slice(
                start_index_axis_pooling_output, stop_index_axis_pooling_output)
            start_index_axis_pooling_output = stop_index_axis_pooling_output

            indexes_combinations[axis_combinations] = c
            indexes_input[axes_removing_input] = (
                combinations_axes_removing_input[tuple(indexes_combinations)])
            array_output[tuple(indexes_output1)] = array_input[tuple(indexes_input)]

            array_output[tuple(indexes_output2)] = combinations_axes_removing_input[tuple(indexes_combinations)]

    return array_output


def merge_neighbouring_conditions(array_input, axis_1, axis_2, m=3, s=None):
    # axis_1 is (are) the axis (axes) that contains (contain) the merging conditions. it can be
    #        an integer or a list of integers. In case of list of integers, the list cannot contain repeated values.
    # axis_2 is the axis that contains the merging samples of a set of n neighbouring conditions in axis_1.
    #        it is always an integer.
    # m is (are) the number(s) of the merging conditions in axis_1. it is an integer is axis_1 is an integer or
    #        a list of integers if axis_1 is a list of integers. In case of list of int, its size has to be the
    #        same as the size of the axis_1.

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = shape_array_input.size

    # format axes_2
    if axis_2 < 0:
        axis_2 += n_axes_array_input

    # format axes_1
    try:
        n_axis_1 = len(axis_1)
        axis_1 = np.asarray(axis_1, dtype=int)
        axis_1[axis_1 < 0] += n_axes_array_input
        # check point 1
        if np.sum(axis_1[0] == axis_1) > 1:
            raise ValueError('axis_1 cannot contain repeated values')
    except TypeError:
        if axis_1 < 0:
            axis_1 += n_axes_array_input
        axis_1 = np.asarray([axis_1], dtype=int)
        n_axis_1 = 1

    axes_merging = np.append(axis_2, axis_1)

    # format m
    try:
        n_m = len(m)
        if n_m == n_axis_1:
            m = np.asarray(m, dtype=int)
        elif n_m == 1:
            m_tmp = m[0]
            m = np.empty(n_axis_1, dtype=int)
            m[:] = m_tmp
            n_m = n_axis_1
        else:
            # check point 3
            raise ValueError('sizes of axis_1 and m must be equal')
    except TypeError:
        m_tmp = m
        m = np.empty(n_axis_1, dtype=int)
        m[:] = m_tmp
        n_m = n_axis_1

    # format s
    if s is None:
        s = m
    else:
        try:
            n_s = len(s)
            if n_s == n_axis_1:
                s = np.asarray(s, dtype=int)
                for a in range(n_axis_1):
                    if s[a] is None:
                        s[a] = m[a]
            elif n_s == 1:
                s_tmp = s[0]
                if s_tmp is None:
                    s = m
                else:
                    s = np.empty(n_axis_1, dtype=int)
                    s[:] = s_tmp
                n_s = n_axis_1
            else:
                # check point 3
                raise ValueError('sizes of axis_1 and s must be equal')
        except TypeError:
            s_tmp = s
            s = np.empty(n_axis_1, dtype=int)
            s[:] = s_tmp
            n_s = n_axis_1

    r_1 = m // 2
    r_0 = r_1
    m_even = (m % 2) == 0
    if any(m_even):
        r_0[m_even] -= 1

    n_conditions_input = shape_array_input[axis_1]
    n_samples_per_condition_input = shape_array_input[axis_2]

    n_conditions_output = ((n_conditions_input - m) // s) + 1
    n_samples_per_m_conditions_output = n_samples_per_condition_input * np.prod(m)

    shape_array_output = shape_array_input
    shape_array_output[axis_1] = n_conditions_output
    shape_array_output[axis_2] = n_samples_per_m_conditions_output
    array_output = np.empty(shape_array_output, dtype=array_input.dtype)

    index_input = np.empty(n_axes_array_input, dtype=object)
    index_input[:] = slice(None)
    index_output = np.copy(index_input)

    center = np.empty(n_axis_1, dtype=object)
    for a in range(n_axis_1):
        center[a] = np.arange(r_0[a], n_conditions_input[a] - r_1[a], s[a])
    combinations_centers = conditions_to_combinations(center)
    n_combinations_centers = combinations_centers.shape[0]

    combinations_merged_conditions_output = n_conditions_to_combinations(n_conditions_output)
    for i in range(n_combinations_centers):
        for a in range(n_axis_1):
            index_input[axis_1[a]] = slice(combinations_centers[i, a] - r_0[a], combinations_centers[i, a] + r_1[a] + 1)
        index_output[axis_1] = combinations_merged_conditions_output[i]

        # array_output[tuple(index_output)] = merge_axes(array_input[tuple(index_input)], axis_1, axis_2)

        array_output[tuple(index_output)] = merge_axes(array_input[tuple(index_input)], axes_merging)

    return array_output
