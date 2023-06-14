import numpy as np
from ..combinations import n_conditions_to_combinations, conditions_to_combinations
from ..array import samples_in_arr1_are_in_arr2


def fast_balanced(
        array_input,
        axes_removing_input=0,
        axis_samples_input=-2,
        axis_variables_table_input=-1,
        variables_inserting_table_output=0,
        variables_staying_table_input=None,
        dtype=None):

    # Input requirements:
    # 1) axes_removing_input cannot contain same values;
    # 2) variables_inserting_table_output cannot contain same values;
    # 3) shapes of axes_removing_input and variables_inserting_table_output must be equal
    # 4) axes_removing_input[a] != axis_samples_input != axis_variables_table_input

    # Import requirements:
    # 1) import numpy as np
    # 2) from ccalafiore.combinations import n_conditions_to_combinations
    # 3) from ccalafiore.array import samples_in_arr1_are_in_arr2
    # 4) from ccalafiore.array import advanced_indexing

    # from ccalafiore.array import samples_in_arr1_are_in_arr2, advanced_indexing

    # format axes_removing_input
    try:
        n_axes_removing_input = len(axes_removing_input)
        axes_removing_input = np.asarray(axes_removing_input, dtype=int)
    except TypeError:
        axes_removing_input = np.asarray([axes_removing_input], dtype=int)
        n_axes_removing_input = len(axes_removing_input)

    # format variables_inserting_in_axes_variables_output
    try:
        n_variables_inserting_table_output = len(variables_inserting_table_output)
        variables_inserting_table_output = np.asarray(variables_inserting_table_output, dtype=int)
    except TypeError:
        variables_inserting_table_output = np.asarray([variables_inserting_table_output], dtype=int)
        n_variables_inserting_table_output = len(variables_inserting_table_output)

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = len(shape_array_input)
    n_variables_table_input = shape_array_input[axis_variables_table_input]
    if axis_variables_table_input < 0:
        axis_variables_table_input += n_axes_array_input
    if axis_samples_input < 0:
        axis_samples_input += n_axes_array_input
    axes_removing_input[axes_removing_input < 0] += n_axes_array_input

    variables_table_input = np.arange(n_variables_table_input)
    if variables_staying_table_input is None:
        variables_staying_table_input = np.copy(variables_table_input)
    n_variables_staying_table_input = len(variables_staying_table_input)
    n_variables_table_output = n_variables_staying_table_input + n_variables_inserting_table_output
    variables_inserting_table_output[variables_inserting_table_output < 0] += n_variables_table_output

    # check point 1
    if np.sum(axes_removing_input[0] == axes_removing_input) > 1:
        raise Exception('axes_removing_input cannot contain repeated values')
    # check point 2
    if np.sum(
            variables_inserting_table_output[0] == variables_inserting_table_output) > 1:
        raise Exception('variables_inserting_table_output cannot contain repeated values')
    # check point 3
    if n_variables_inserting_table_output != n_axes_removing_input:
        raise Exception(
            'Shapes of axes_removing_input and variables_inserting_table_output must be equal')

    # check point 4
    if np.sum(axes_removing_input == axis_samples_input) > 0:
        raise Exception('axes_removing_input[a] != axis_samples_input')
    if np.sum(axes_removing_input == axis_variables_table_input) > 0:
        raise Exception('axes_removing_input[a] != axis_variables_table_input')
    if np.sum(axis_samples_input == axis_variables_table_input) > 0:
        raise Exception('axis_samples_input != axis_variables_table_input')

    axes_array_input = np.arange(n_axes_array_input)
    axes_other_array_input = axes_array_input[axes_array_input != axis_variables_table_input]
    axes_other_array_input = axes_other_array_input[
        axes_other_array_input != axis_samples_input]
    axes_other_array_input = axes_other_array_input[np.logical_not(
        samples_in_arr1_are_in_arr2(axes_other_array_input, axes_removing_input))]
    # axes_other_array_input_inverted = axes_other_array_input[::-1]
    # n_axes_other_array_input = len(axes_other_array_input)

    axis_variables_table_output = axis_variables_table_input - np.sum(axes_removing_input < axis_variables_table_input)
    axis_samples_output = axis_samples_input - np.sum(axes_removing_input < axis_samples_input)
    # changed = True
    # while changed:
    #     changed = False
    #     while axis_variables_table_output in axes_removing_input:
    #         axis_variables_table_output += 1
    #         changed = True
    #     while axis_samples_output in axes_removing_input:
    #         axis_samples_output += 1
    #         changed = True
    #     if axis_variables_table_output == axis_samples_output:
    #         changed = True
    #         if axis_variables_table_input > axis_samples_input:
    #             axis_variables_table_output += 1
    #         elif axis_variables_table_input < axis_samples_input:
    #             axis_samples_output += 1

    n_axes_array_output = n_axes_array_input - n_axes_removing_input
    axes_array_output = np.arange(n_axes_array_output)
    axes_non_axis_variables_table_output = axes_array_output[axes_array_output != axis_variables_table_output]
    axes_other_output = axes_non_axis_variables_table_output[
        axes_non_axis_variables_table_output != axis_samples_output]

    shape_array_output = np.empty(n_axes_array_output, dtype=object)
    n_conditions_in_axes_removing_input = shape_array_input[axes_removing_input]
    shape_array_output[axis_samples_output] = (
            shape_array_input[axis_samples_input] * np.prod(n_conditions_in_axes_removing_input))
    shape_array_output[axis_variables_table_output] = n_variables_table_output
    shape_array_output[axes_other_output] = shape_array_input[axes_other_array_input]

    if dtype is None:
        dtype = array_input.dtype
    array_output = np.empty(shape_array_output, dtype=dtype)

    variables_table_output = np.arange(n_variables_table_output)
    variables_staying_table_output = variables_table_output[np.logical_not(
        samples_in_arr1_are_in_arr2(variables_table_output, variables_inserting_table_output))]

    axis_variables_in_combinations_removing = int(axis_variables_table_input > axis_samples_input)
    axis_combinations_in_combinations_removing = int(not (bool(axis_variables_in_combinations_removing)))
    combinations_axes_removing_input = n_conditions_to_combinations(
        n_conditions_in_axes_removing_input, axis_combinations=axis_combinations_in_combinations_removing)
    n_combinations_axes_removing_input = (
        combinations_axes_removing_input.shape[axis_combinations_in_combinations_removing])

    indexes_combinations_removing_input = np.empty(2, dtype=object)
    indexes_combinations_removing_input[axis_variables_in_combinations_removing] = slice(None)

    indexes_combinations_removing_output = np.empty(2, dtype=object)
    indexes_combinations_removing_output[axis_variables_in_combinations_removing] = np.arange(n_axes_removing_input)

    # for a in axes_other_output:
    #     combinations_axes_removing_input = np.expand_dims(combinations_axes_removing_input, axis=a)



    axes_removing_input_inverted = np.sort(axes_removing_input)[::-1]

    # index_input = np.full(n_axes_array_input, slice(None), dtype=object)
    indexes_input = np.empty(n_axes_array_input, dtype=object)
    for a in range(n_axes_array_input):
        if (a not in axes_removing_input) and (a != axis_variables_table_input):
            indexes_input[a] = np.arange(shape_array_input[a])

    indexes_input[axis_variables_table_input] = variables_staying_table_input

    # index_array_output = np.full(n_axes_array_output, slice(None), dtype=object)
    index_array_output = np.empty(n_axes_array_output, dtype=object)
    for a in range(n_axes_array_output):
        if (a != axis_variables_table_output) and (a != axis_samples_output):
            index_array_output[a] = np.arange(shape_array_output[a])

    length_axis_flattering_input = shape_array_input[axis_samples_input]

    start_index_axis_samples_output = 0
    for c in range(n_combinations_axes_removing_input):

        stop_index_axis_samples_output = (c + 1) * length_axis_flattering_input
        index_array_output[axis_samples_output] = np.arange(
            start_index_axis_samples_output, stop_index_axis_samples_output)
        start_index_axis_samples_output = stop_index_axis_samples_output

        index_array_output[axis_variables_table_output] = variables_staying_table_output

        indexes_combinations_removing_input[axis_combinations_in_combinations_removing] = c
        indexes_input[axes_removing_input] = (
            combinations_axes_removing_input[tuple(indexes_combinations_removing_input)])
        array_input_c = array_input[advanced_indexing(indexes_input)]
        for a in axes_removing_input_inverted:
            array_input_c = np.squeeze(array_input_c, axis=a)

        array_output[advanced_indexing(index_array_output)] = array_input_c
        index_array_output[axis_variables_table_output] = variables_inserting_table_output

        indexes_combinations_removing_output[axis_combinations_in_combinations_removing] = c
        combinations_axes_removing_input_c = combinations_axes_removing_input[
            advanced_indexing(indexes_combinations_removing_output)]
        for a in axes_other_output:
            combinations_axes_removing_input_c = np.expand_dims(combinations_axes_removing_input_c, axis=a)

        array_output[advanced_indexing(index_array_output)] = combinations_axes_removing_input_c

    return array_output
