import numpy as np
from ...array import samples_in_arr1_are_in_arr2  # , advanced_indexing
from ...combinations import (
    trials_to_conditions, conditions_to_n_conditions, n_conditions_to_combinations, conditions_to_combinations)


def from_2_arrays(
        array_variables_staying,
        array_variables_adding_axes,
        axis_samples,
        axis_variables_table,
        axes_inserting=None,
        dtype=None,
        format_and_check=True):

    axis_variables_table_input = axis_variables_table
    axis_samples_input = axis_samples

    shape_array_variables_staying = np.asarray(array_variables_staying.shape)
    n_axes_array_input = shape_array_variables_staying.size
    n_variables_table_staying = shape_array_variables_staying[axis_variables_table_input]

    if format_and_check:
        n_axes_array_variables_staying = n_axes_array_input

        shape_array_variables_adding_axes = np.asarray(array_variables_adding_axes.shape)
        n_axes_array_variables_adding_axes = shape_array_variables_adding_axes.size

        # check point 1
        if n_axes_array_variables_staying != n_axes_array_variables_adding_axes:
            raise ValueError('dimension mismatch')

        if axis_variables_table_input < 0:
            axis_variables_table_input += n_axes_array_input
        if axis_samples_input < 0:
            axis_samples_input += n_axes_array_input

        # check point 5
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        # check point 2
        indexes_logical = np.arange(n_axes_array_input) != axis_variables_table_input
        if np.any(shape_array_variables_staying[indexes_logical] != shape_array_variables_adding_axes[indexes_logical]):
            raise ValueError('dimension mismatch')

        n_variables_table_adding_axes = shape_array_variables_adding_axes[axis_variables_table_input]

        # format axes_inserting
        n_axes_inserting = n_variables_table_adding_axes
        n_axes_array_output = n_axes_array_input + n_axes_inserting
        try:
            len(axes_inserting)
            axes_inserting = np.asarray(axes_inserting, dtype=int)
            axes_inserting[axes_inserting < 0] += n_axes_array_output
            # check point 3
            if np.sum(axes_inserting[0] == axes_inserting) > 1:
                raise ValueError('axes_inserting cannot contain repeated values')
        except TypeError:
            if axes_inserting is None:
                axes_inserting = np.arange(n_axes_inserting)
            else:
                if axes_inserting < 0:
                    axes_inserting += n_axes_array_output
                axes_inserting = np.asarray([axes_inserting], dtype=int)

    else:
        n_axes_inserting = axes_inserting.size
        n_axes_array_output = n_axes_array_input + n_axes_inserting

    axes_array_input = np.arange(n_axes_array_input)
    axes_non_axis_variables_table_input = axes_array_input[axes_array_input != axis_variables_table_input]
    axes_other_array_input = axes_non_axis_variables_table_input[
        axes_non_axis_variables_table_input != axis_samples_input]
    n_axes_other_array_input = axes_other_array_input.size

    indexes_array_variables_staying = np.empty(n_axes_array_input, dtype=object)
    indexes_array_variables_staying[:] = slice(None)

    axis_variables_in_combinations = int(axis_variables_table_input > axis_samples_input)
    axis_combinations_in_combinations = int(not (bool(axis_variables_in_combinations)))

    if n_axes_other_array_input == 0:
        conditions_variables_table_adding_axes = trials_to_conditions(
            array_variables_adding_axes, axis_combinations=axis_combinations_in_combinations)
    elif n_axes_other_array_input > 0:
        indexes_array_variables_adding_axes = np.copy(indexes_array_variables_staying)
        indexes_array_variables_adding_axes[axes_other_array_input] = 0
        array_variables_adding_axes_1 = array_variables_adding_axes[tuple(indexes_array_variables_adding_axes)]
        conditions_variables_table_adding_axes = trials_to_conditions(
            array_variables_adding_axes_1, axis_combinations=axis_combinations_in_combinations)

    n_conditions_variables_table_adding_axes = conditions_to_n_conditions(
        conditions_variables_table_adding_axes)
    n_combinations_variables_table_adding_axes = int(np.prod(
        n_conditions_variables_table_adding_axes))

    combinations_variables_table_adding_axes = conditions_to_combinations(
        conditions_variables_table_adding_axes,
        axis_combinations=axis_combinations_in_combinations)

    combinations_axes_inserting = n_conditions_to_combinations(
        n_conditions_variables_table_adding_axes)

    axis_variables_table_output = axis_variables_table_input + np.sum(
        axes_inserting <= axis_variables_table_input)
    axis_samples_output = axis_samples_input + np.sum(axes_inserting <= axis_samples_input)
    need_to_check = True
    while need_to_check:
        need_to_check = False
        while axis_variables_table_output in axes_inserting:
            axis_variables_table_output += 1
            need_to_check = True
        while axis_samples_output in axes_inserting:
            axis_samples_output += 1
            need_to_check = True
        if axis_variables_table_output == axis_samples_output:
            if axis_variables_table_input > axis_samples_input:
                axis_variables_table_output += 1
            elif axis_variables_table_input < axis_samples_input:
                axis_samples_output += 1
            need_to_check = True

    axes_array_output = np.arange(n_axes_array_output)
    axes_non_axis_variables_table_output = axes_array_output[axes_array_output != axis_variables_table_output]
    axes_other_output = axes_non_axis_variables_table_output[
        axes_non_axis_variables_table_output != axis_samples_output]
    axes_other_output = axes_other_output[np.logical_not(
        samples_in_arr1_are_in_arr2(axes_other_output, axes_inserting))]
    # axes_inserting_sorted = np.sort(axes_inserting)

    shape_array_output = np.empty(n_axes_array_output, dtype=int)
    shape_array_output[axes_inserting] = n_conditions_variables_table_adding_axes
    shape_array_output[axis_variables_table_output] = n_variables_table_staying
    shape_array_output[axis_samples_output] = (
            shape_array_variables_staying[axis_samples_input] / n_combinations_variables_table_adding_axes)
    shape_array_output[axes_other_output] = shape_array_variables_staying[axes_other_array_input]
    if dtype is None:
        dtype = array_variables_staying.dtype
    array_output = np.empty(shape_array_output, dtype=dtype)

    indexes_output = np.empty(n_axes_array_output, dtype=object)
    indexes_output[:] = slice(None)

    indexes_combinations = np.empty(2, dtype=object)
    indexes_combinations[axis_variables_in_combinations] = slice(None)

    if n_axes_other_array_input == 0:
        for i in range(n_combinations_variables_table_adding_axes):
            indexes_output[axes_inserting] = combinations_axes_inserting[i]
            indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
            indexes_array_variables_staying[axis_samples_input] = np.all(
                array_variables_adding_axes ==
                combinations_variables_table_adding_axes[tuple(indexes_combinations)],
                axis=axis_variables_in_combinations)
            array_output[tuple(indexes_output)] = array_variables_staying[tuple(
                indexes_array_variables_staying)]

    elif n_axes_other_array_input > 0:
        indexes_array_variables_staying_j = np.empty(2, dtype=object)
        indexes_array_variables_staying_j[axis_variables_in_combinations] = slice(None)

        combinations_axes_other = n_conditions_to_combinations(shape_array_variables_staying[axes_other_array_input])
        n_combinations_axes_other = combinations_axes_other.shape[0]
        for j in range(n_combinations_axes_other):
            combinations_axes_other_j = combinations_axes_other[j]
            indexes_output[axes_other_output] = combinations_axes_other_j
            indexes_array_variables_staying[axes_other_array_input] = combinations_axes_other_j
            indexes_array_variables_adding_axes[axes_other_array_input] = combinations_axes_other_j
            array_variables_adding_axes_j = array_variables_adding_axes[
                tuple(indexes_array_variables_adding_axes)]
            array_variables_staying_j = array_variables_staying[tuple(
                indexes_array_variables_staying)]
            for i in range(n_combinations_variables_table_adding_axes):
                indexes_output[axes_inserting] = combinations_axes_inserting[i]
                indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                indexes_array_variables_staying_j[axis_combinations_in_combinations] = np.all(
                    array_variables_adding_axes_j ==
                    combinations_variables_table_adding_axes[tuple(indexes_combinations)],
                    axis=axis_variables_in_combinations)
                array_output[tuple(indexes_output)] = array_variables_staying_j[tuple(
                    indexes_array_variables_staying_j)]

    return array_output


def from_2_arrays_advanced(
        array_variables_staying,
        array_variables_adding_axes,
        axis_samples,
        axis_variables_table,
        variables_table_adding_axes=None,
        axes_inserting=None,
        variables_table_staying=None,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) uses the efficient function slice for indexes;
    # 2) it splits the array in two arrays first: array_variables_adding_axes and array_variables_staying.
    #    Then, it makes the array output from those two arrays;
    # 3) it assumes that the numbers of samples (or trials) in the axis_samples for all combinations
    #    of variables_table_adding_axes are equal;
    # 4) it does not assumes that the order of samples (or trials) in the axis_samples are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_variables_table. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) the numbers of axes of the two arrays must be equal;
    # 2) the shapes of the 2 arrays must be equal, except for the axis_variable_table,
    #    i.e. the axis that contains the variables of the tables;
    # 3) axes_inserting cannot contain repeated values;
    # 4) variables_table_adding_axes cannot contain repeated values;
    # 5) shapes of axes_inserting and variables_table_adding_axes must be equal
    # 6) axis_samples != axis_variables_table
    # 7) the numbers of samples (or trials) in the axis_samples for all combinations
    #    of variables_table_adding_axes are equal

    # Import requirements:
    # 1) import numpy as np
    # 2) from ccalafiore.combinations import n_conditions_to_combinations
    # 3) from ccalafiore.combinations import trials_to_conditions
    # 4) from ccalafiore.combinations import conditions_to_n_conditions
    # 5) from ccalafiore.combinations import conditions_to_combinations
    # 6) from ccalafiore.array import samples_in_arr1_are_in_arr2

    # from ccalafiore.combinations import \
    #     trials_to_conditions, conditions_to_n_conditions, n_conditions_to_combinations, conditions_to_combinations
    # from ccalafiore.array import samples_in_arr1_are_in_arr2

    axis_variables_table_input = axis_variables_table
    axis_samples_input = axis_samples

    shape_array_variables_staying = np.asarray(array_variables_staying.shape)
    n_axes_array_variables_staying = shape_array_variables_staying.size
    n_variables_table_in_array_variables_staying = shape_array_variables_staying[axis_variables_table_input]
    variables_table_in_array_variables_staying = np.arange(n_variables_table_in_array_variables_staying)

    shape_array_variables_adding_axes = np.asarray(array_variables_adding_axes.shape)
    n_axes_array_variables_adding_axes = shape_array_variables_adding_axes.size
    n_variables_table_in_array_variables_adding_axes = shape_array_variables_adding_axes[axis_variables_table_input]
    variables_table_in_array_variables_adding_axes = np.arange(n_variables_table_in_array_variables_adding_axes)

    if format_and_check:
        # check point 1
        if n_axes_array_variables_staying != n_axes_array_variables_adding_axes:
            raise ValueError('dimension mismatch')
        else:
            n_axes_array_input = n_axes_array_variables_staying

        if axis_variables_table_input < 0:
            axis_variables_table_input += n_axes_array_input
        if axis_samples_input < 0:
            axis_samples_input += n_axes_array_input

        # check point 6
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        # check point 2
        indexes_logical = np.arange(n_axes_array_input) != axis_variables_table_input
        if np.any(shape_array_variables_staying[indexes_logical] != shape_array_variables_adding_axes[indexes_logical]):
            raise ValueError('dimension mismatch')

        n_variables_table_in_array_variables_staying = shape_array_variables_staying[axis_variables_table_input]
        variables_table_in_array_variables_staying = np.arange(n_variables_table_in_array_variables_staying)

        n_variables_table_in_array_variables_adding_axes = shape_array_variables_adding_axes[axis_variables_table_input]
        variables_table_in_array_variables_adding_axes = np.arange(n_variables_table_in_array_variables_adding_axes)

        # format variables_table_adding_axes
        try:
            n_variables_table_adding_axes = len(variables_table_adding_axes)
            variables_table_adding_axes = np.asarray(variables_table_adding_axes, dtype=int)
            variables_table_adding_axes[variables_table_adding_axes < 0] += \
                n_variables_table_in_array_variables_adding_axes
            # check point 4
            if np.sum(variables_table_adding_axes[0] == variables_table_adding_axes) > 1:
                raise ValueError('variables_table_adding_axes cannot contain repeated values')
        except TypeError:
            if variables_table_adding_axes is None:
                variables_table_adding_axes = np.arange(n_variables_table_in_array_variables_adding_axes)
                n_variables_table_adding_axes = n_variables_table_in_array_variables_adding_axes
            else:
                if variables_table_adding_axes < 0:
                    variables_table_adding_axes += n_variables_table_in_array_variables_adding_axes
                variables_table_adding_axes = np.asarray([variables_table_adding_axes], dtype=int)
                n_variables_table_adding_axes = 1

        # format variables_table_staying
        try:
            n_variables_table_staying = len(variables_table_staying)
            variables_table_staying = np.asarray(variables_table_staying, dtype=int)
            variables_table_staying[variables_table_staying < 0] += n_variables_table_in_array_variables_staying
        except TypeError:
            if variables_table_staying is None:
                variables_table_staying = np.arange(n_variables_table_in_array_variables_staying)
                n_variables_table_staying = n_variables_table_in_array_variables_staying
            else:
                if variables_table_staying < 0:
                    variables_table_staying += n_variables_table_in_array_variables_staying
                variables_table_staying = np.asarray([variables_table_staying], dtype=int)
                n_variables_table_staying = 1

        # format axes_inserting
        n_axes_inserting = n_variables_table_adding_axes
        n_axes_array_output = n_axes_array_input + n_axes_inserting
        try:
            n_axes_inserting = len(axes_inserting)
            axes_inserting = np.asarray(axes_inserting, dtype=int)
            axes_inserting[axes_inserting < 0] += n_axes_array_output
            # check point 3
            if np.sum(axes_inserting[0] == axes_inserting) > 1:
                raise ValueError('axes_inserting cannot contain repeated values')
            # check point 5
            if n_variables_table_adding_axes != n_axes_inserting:
                raise ValueError('Shapes of axes_inserting and variables_table_adding_axes must be equal')
        except TypeError:
            if axes_inserting is None:
                axes_inserting = np.arange(n_axes_inserting)
            else:
                if axes_inserting < 0:
                    axes_inserting += n_axes_array_output
                axes_inserting = np.asarray([axes_inserting], dtype=int)
                n_axes_inserting = 1
                # check point 5
                if n_variables_table_adding_axes != n_axes_inserting:
                    raise ValueError('Shapes of axes_inserting and variables_table_adding_axes must be equal')
    else:
        n_variables_table_staying = variables_table_staying.size
        n_variables_table_adding_axes = variables_table_adding_axes.size

    indexes = None
    if ((n_variables_table_staying != n_variables_table_in_array_variables_staying) or
            np.any(variables_table_staying != variables_table_in_array_variables_staying)):

        indexes = np.empty(n_axes_array_variables_staying, dtype=object)
        indexes[:] = slice(None)
        indexes[axis_variables_table_input] = variables_table_staying
        array_variables_staying = array_variables_staying[tuple(indexes)]

    if ((n_variables_table_adding_axes != n_variables_table_in_array_variables_adding_axes) or
            np.any(variables_table_adding_axes != variables_table_in_array_variables_adding_axes)):

        if indexes is None:
            indexes = np.empty(n_axes_array_variables_adding_axes, dtype=object)
            indexes[:] = slice(None)
        indexes[axis_variables_table_input] = variables_table_adding_axes
        array_variables_adding_axes = array_variables_adding_axes[tuple(indexes)]

    array_output = from_2_arrays(
        array_variables_staying,
        array_variables_adding_axes,
        axis_samples_input,
        axis_variables_table_input,
        axes_inserting=axes_inserting,
        dtype=dtype,
        format_and_check=False)

    return array_output


def from_1_array(
        array_input,
        axis_samples,
        axis_variables_table,
        variables_table_adding_axes,
        axes_inserting=None,
        variables_table_staying=None,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) uses the efficient function slice for indexes;
    # 2) it splits the array in two arrays first: array_variables_adding_axes and array_variables_staying.
    #    Then, it makes the array output from those two arrays;
    # 3) the numbers of samples (or trials) in the axis_samples for all combinations
    #    of variables_table_adding_axes have to be equal;
    # 4) it does not assumes that the order of samples (or trials) in the axis_samples are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_variables_table. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) axes_inserting cannot contain repeated values;
    # 2) variables_table_adding_axes cannot contain repeated values;
    # 3) shapes of axes_inserting and variables_table_adding_axes must be equal
    # 4) axis_samples != axis_variables_table
    # 5) the numbers of samples (or trials) in the axis_samples for all combinations
    #    of variables_table_adding_axes are equal

    # Import requirements:
    # 1) import numpy as np
    # 2) from ccalafiore.combinations import n_conditions_to_combinations
    # 3) from ccalafiore.combinations import trials_to_conditions
    # 4) from ccalafiore.combinations import conditions_to_n_conditions
    # 5) from ccalafiore.combinations import conditions_to_combinations
    # 6) from ccalafiore.array import samples_in_arr1_are_in_arr2

    # from ccalafiore.combinations import \
    #     trials_to_conditions, conditions_to_n_conditions, n_conditions_to_combinations, conditions_to_combinations
    # from ccalafiore.array import samples_in_arr1_are_in_arr2

    axis_variables_table_input = axis_variables_table
    axis_samples_input = axis_samples

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = shape_array_input.size
    n_variables_table_input = shape_array_input[axis_variables_table_input]
    variables_table_input = np.arange(n_variables_table_input)

    if format_and_check:
        if axis_variables_table_input < 0:
            axis_variables_table_input += n_axes_array_input
        if axis_samples_input < 0:
            axis_samples_input += n_axes_array_input

        # check point 4
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        # format variables_table_adding_axes
        try:
            n_variables_table_adding_axes = len(variables_table_adding_axes)
            variables_table_adding_axes = np.asarray(variables_table_adding_axes, dtype=int)
            variables_table_adding_axes[variables_table_adding_axes < 0] += n_variables_table_input
            # check point 2
            if np.sum(variables_table_adding_axes[0] == variables_table_adding_axes) > 1:
                raise ValueError('variables_table_adding_axes cannot contain repeated values')
        except TypeError:
            if variables_table_adding_axes < 0:
                variables_table_adding_axes += n_variables_table_input
            variables_table_adding_axes = np.asarray([variables_table_adding_axes], dtype=int)
            n_variables_table_adding_axes = 1

        # format variables_table_staying
        try:
            n_variables_table_staying = len(variables_table_staying)
            variables_table_staying = np.asarray(variables_table_staying, dtype=int)
            variables_table_staying[variables_table_staying < 0] += n_variables_table_input
        except TypeError:
            if variables_table_staying is None:
                variables_table_staying = variables_table_input[np.logical_not(
                    samples_in_arr1_are_in_arr2(variables_table_input, variables_table_adding_axes))]
                n_variables_table_staying = variables_table_staying.size
            else:
                if variables_table_staying < 0:
                    variables_table_staying += n_variables_table_input
                variables_table_staying = np.asarray([variables_table_staying], dtype=int)
                n_variables_table_staying = 1

        # format axes_inserting
        n_axes_inserting = n_variables_table_adding_axes
        n_axes_array_output = n_axes_array_input + n_axes_inserting
        try:
            n_axes_inserting = len(axes_inserting)
            axes_inserting = np.asarray(axes_inserting, dtype=int)
            axes_inserting[axes_inserting < 0] += n_axes_array_output
            # check point 1
            if np.sum(axes_inserting[0] == axes_inserting) > 1:
                raise ValueError('axes_inserting cannot contain repeated values')
            # check point 3
            if n_variables_table_adding_axes != n_axes_inserting:
                raise ValueError('Shapes of axes_inserting and variables_table_adding_axes must be equal')
        except TypeError:
            if axes_inserting is None:
                axes_inserting = np.arange(n_axes_inserting)
            else:
                if axes_inserting < 0:
                    axes_inserting += n_axes_array_output
                axes_inserting = np.asarray([axes_inserting], dtype=int)
                n_axes_inserting = 1
                # check point 3
                if n_variables_table_adding_axes != n_axes_inserting:
                    raise ValueError('Shapes of axes_inserting and variables_table_adding_axes must be equal')
    else:
        n_variables_table_staying = variables_table_staying.size
        n_variables_table_adding_axes = variables_table_adding_axes.size

    indexes = np.empty(n_axes_array_input, dtype=object)
    indexes[:] = slice(None)
    if ((n_variables_table_staying != n_variables_table_input) or
            np.any(variables_table_staying != variables_table_input)):
        indexes[axis_variables_table_input] = variables_table_staying
    array_variables_staying = array_input[tuple(indexes)]

    if ((n_variables_table_adding_axes != n_variables_table_input) or
            np.any(variables_table_adding_axes != variables_table_input)):
        indexes[axis_variables_table_input] = variables_table_adding_axes
    else:
        indexes[axis_variables_table_input] = slice(None)
    array_variables_adding_axes = array_input[tuple(indexes)]
    del array_input

    array_output = from_2_arrays(
        array_variables_staying,
        array_variables_adding_axes,
        axis_samples_input,
        axis_variables_table_input,
        axes_inserting=axes_inserting,
        dtype=dtype,
        format_and_check=False)

    return array_output
