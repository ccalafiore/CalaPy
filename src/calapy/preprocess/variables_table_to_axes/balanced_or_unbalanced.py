import numpy as np
from ...array import samples_in_arr1_are_in_arr2  # , advanced_indexing
from ...combinations import (
    trials_to_conditions, conditions_to_n_conditions,
    n_conditions_to_combinations, conditions_to_combinations)
from ..reshape import merge_axes_and_add_axes_removing_info_to_axis_receiving


def from_2_arrays(
        array_data_staying,
        array_indexes_adding_axes,
        axis_spitting_input,
        axis_indexes_adding_axes_input,
        axes_inserting_output=None,
        balance=True,
        dtype=None,
        format_and_check=True):

    shape_array_data_staying = np.asarray(array_data_staying.shape)
    n_axes_array_input = shape_array_data_staying.size

    if balance:

        n_indexes_staying = shape_array_data_staying[axis_indexes_adding_axes_input]

        if format_and_check:
            n_axes_array_data_staying = n_axes_array_input

            shape_array_indexes_adding_axes = np.asarray(array_indexes_adding_axes.shape)
            n_axes_array_indexes_adding_axes = shape_array_indexes_adding_axes.size

            # check point 1
            if n_axes_array_data_staying != n_axes_array_indexes_adding_axes:
                raise ValueError('dimension mismatch')

            if axis_indexes_adding_axes_input < 0:
                axis_indexes_adding_axes_input += n_axes_array_input
            if axis_spitting_input < 0:
                axis_spitting_input += n_axes_array_input

            # check point 5
            if axis_spitting_input == axis_indexes_adding_axes_input:
                raise ValueError('axis_spitting_input and axis_indexes_adding_axes_input must be different')

            # check point 2
            indexes_logical = np.arange(n_axes_array_input) != axis_indexes_adding_axes_input
            if np.any(shape_array_data_staying[indexes_logical] !=
                      shape_array_indexes_adding_axes[indexes_logical]):
                raise ValueError('dimension mismatch')

            n_indexes_table_adding_axes = shape_array_indexes_adding_axes[axis_indexes_adding_axes_input]

            # format axes_inserting_output
            n_axes_inserting_output = n_indexes_table_adding_axes
            n_axes_array_output = n_axes_array_input + n_axes_inserting_output
            try:
                len(axes_inserting_output)
                axes_inserting_output = np.asarray(axes_inserting_output, dtype=int)
                axes_inserting_output[axes_inserting_output < 0] += n_axes_array_output
                # check point 3
                for a in axes_inserting_output:
                    if np.sum(a == axes_inserting_output) > 1:
                        raise ValueError('axes_inserting_output cannot contain repeated values')
            except TypeError:
                if axes_inserting_output is None:
                    axes_inserting_output = np.arange(n_axes_inserting_output)
                else:
                    if axes_inserting_output < 0:
                        axes_inserting_output += n_axes_array_output
                    axes_inserting_output = np.asarray([axes_inserting_output], dtype=int)

        else:
            n_axes_inserting_output = axes_inserting_output.size
            n_axes_array_output = n_axes_array_input + n_axes_inserting_output

        axes_array_input = np.arange(n_axes_array_input)
        axes_non_axis_indexes_adding_axes_input = axes_array_input[axes_array_input != axis_indexes_adding_axes_input]
        axes_other_array_input = axes_non_axis_indexes_adding_axes_input[
            axes_non_axis_indexes_adding_axes_input != axis_spitting_input]
        n_axes_other_array_input = axes_other_array_input.size

        indexes_array_data_staying = np.empty(n_axes_array_input, dtype=object)
        indexes_array_data_staying[:] = slice(None)

        axis_variables_in_combinations = int(axis_indexes_adding_axes_input > axis_spitting_input)
        axis_combinations_in_combinations = int(not (bool(axis_variables_in_combinations)))

        if n_axes_other_array_input == 0:
            conditions_indexes_table_adding_axes = trials_to_conditions(
                array_indexes_adding_axes, axis_combinations=axis_combinations_in_combinations)
        elif n_axes_other_array_input > 0:
            indexes_array_indexes_adding_axes = np.copy(indexes_array_data_staying)
            indexes_array_indexes_adding_axes[axes_other_array_input] = 0
            array_indexes_adding_axes_1 = array_indexes_adding_axes[tuple(indexes_array_indexes_adding_axes)]
            conditions_indexes_table_adding_axes = trials_to_conditions(
                array_indexes_adding_axes_1, axis_combinations=axis_combinations_in_combinations)

        n_conditions_indexes_table_adding_axes = conditions_to_n_conditions(
            conditions_indexes_table_adding_axes)
        n_combinations_indexes_table_adding_axes = int(np.prod(
            n_conditions_indexes_table_adding_axes))

        combinations_indexes_table_adding_axes = conditions_to_combinations(
            conditions_indexes_table_adding_axes,
            axis_combinations=axis_combinations_in_combinations)

        combinations_axes_inserting_output = n_conditions_to_combinations(
            n_conditions_indexes_table_adding_axes)

        axis_indexes_removing_output = axis_indexes_adding_axes_input + np.sum(
            axes_inserting_output <= axis_indexes_adding_axes_input)
        axis_spitting_output = axis_spitting_input + np.sum(axes_inserting_output <= axis_spitting_input)
        need_to_check = True
        while need_to_check:
            need_to_check = False
            while axis_indexes_removing_output in axes_inserting_output:
                axis_indexes_removing_output += 1
                need_to_check = True
            while axis_spitting_output in axes_inserting_output:
                axis_spitting_output += 1
                need_to_check = True
            if axis_indexes_removing_output == axis_spitting_output:
                if axis_indexes_adding_axes_input > axis_spitting_input:
                    axis_indexes_removing_output += 1
                elif axis_indexes_adding_axes_input < axis_spitting_input:
                    axis_spitting_output += 1
                need_to_check = True

        axes_array_output = np.arange(n_axes_array_output)
        axes_non_axis_indexes_removing_output = axes_array_output[axes_array_output != axis_indexes_removing_output]
        axes_other_output = axes_non_axis_indexes_removing_output[
            axes_non_axis_indexes_removing_output != axis_spitting_output]
        axes_other_output = axes_other_output[np.logical_not(
            samples_in_arr1_are_in_arr2(axes_other_output, axes_inserting_output))]
        # axes_inserting_output_sorted = np.sort(axes_inserting_output)

        shape_array_output = np.empty(n_axes_array_output, dtype=int)
        shape_array_output[axes_inserting_output] = n_conditions_indexes_table_adding_axes
        shape_array_output[axis_indexes_removing_output] = n_indexes_staying
        shape_array_output[axis_spitting_output] = (
                shape_array_data_staying[axis_spitting_input] / n_combinations_indexes_table_adding_axes)
        shape_array_output[axes_other_output] = shape_array_data_staying[axes_other_array_input]
        if dtype is None:
            dtype = array_data_staying.dtype
        array_output = np.empty(shape_array_output, dtype=dtype)

        indexes_output = np.empty(n_axes_array_output, dtype=object)
        indexes_output[:] = slice(None)

        indexes_combinations = np.empty(2, dtype=object)
        indexes_combinations[axis_variables_in_combinations] = slice(None)

        if n_axes_other_array_input == 0:
            for i in range(n_combinations_indexes_table_adding_axes):
                indexes_output[axes_inserting_output] = combinations_axes_inserting_output[i]
                indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                indexes_array_data_staying[axis_spitting_input] = np.all(
                    array_indexes_adding_axes ==
                    combinations_indexes_table_adding_axes[tuple(indexes_combinations)],
                    axis=axis_variables_in_combinations)
                array_output[tuple(indexes_output)] = array_data_staying[tuple(
                    indexes_array_data_staying)]

        elif n_axes_other_array_input > 0:
            indexes_array_data_staying_j = np.empty(2, dtype=object)
            indexes_array_data_staying_j[axis_variables_in_combinations] = slice(None)

            combinations_axes_other = n_conditions_to_combinations(
                shape_array_data_staying[axes_other_array_input])
            n_combinations_axes_other = combinations_axes_other.shape[0]
            for j in range(n_combinations_axes_other):
                combinations_axes_other_j = combinations_axes_other[j]
                indexes_output[axes_other_output] = combinations_axes_other_j
                indexes_array_data_staying[axes_other_array_input] = combinations_axes_other_j
                indexes_array_indexes_adding_axes[axes_other_array_input] = combinations_axes_other_j
                array_indexes_adding_axes_j = array_indexes_adding_axes[
                    tuple(indexes_array_indexes_adding_axes)]
                array_data_staying_j = array_data_staying[tuple(
                    indexes_array_data_staying)]
                for i in range(n_combinations_indexes_table_adding_axes):
                    indexes_output[axes_inserting_output] = combinations_axes_inserting_output[i]
                    indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                    indexes_array_data_staying_j[axis_combinations_in_combinations] = np.all(
                        array_indexes_adding_axes_j ==
                        combinations_indexes_table_adding_axes[tuple(indexes_combinations)],
                        axis=axis_variables_in_combinations)
                    array_output[tuple(indexes_output)] = array_data_staying_j[tuple(
                        indexes_array_data_staying_j)]

    else:
        # n_indexes_staying = shape_array_data_staying[axis_indexes_adding_axes_input]
        axes_array_input = np.arange(n_axes_array_input)
        axes_non_axis_indexes_adding_axes_input = axes_array_input[axes_array_input != axis_indexes_adding_axes_input]
        axes_other_array_input = axes_non_axis_indexes_adding_axes_input[
            axes_non_axis_indexes_adding_axes_input != axis_spitting_input]
        n_axes_other_array_input = axes_other_array_input.size

        if format_and_check:
            n_axes_array_data_staying = n_axes_array_input

            shape_array_indexes_adding_axes = np.asarray(array_indexes_adding_axes.shape)
            n_axes_array_indexes_adding_axes = shape_array_indexes_adding_axes.size

            # check point 1
            if n_axes_array_data_staying != n_axes_array_indexes_adding_axes:
                raise ValueError('dimension mismatch')

            if axis_indexes_adding_axes_input < 0:
                axis_indexes_adding_axes_input += n_axes_array_input
            if axis_spitting_input < 0:
                axis_spitting_input += n_axes_array_input

            # check point 5
            if axis_spitting_input == axis_indexes_adding_axes_input:
                raise ValueError('axis_spitting_input and axis_indexes_adding_axes_input must be different')

            # check point 2
            indexes_logical = np.arange(n_axes_array_input) != axis_indexes_adding_axes_input
            if np.any(shape_array_data_staying[indexes_logical] !=
                      shape_array_indexes_adding_axes[indexes_logical]):
                raise ValueError('dimension mismatch')

            n_indexes_table_adding_axes = shape_array_indexes_adding_axes[axis_indexes_adding_axes_input]

            # format axes_inserting_output
            n_axes_inserting_output = n_indexes_table_adding_axes
            n_axes_array_output_object = n_axes_inserting_output
            try:
                len(axes_inserting_output)
                axes_inserting_output = np.asarray(axes_inserting_output, dtype=int)
                axes_inserting_output[axes_inserting_output < 0] += n_axes_array_output_object
                # check point 3
                for a in axes_inserting_output:
                    if np.sum(a == axes_inserting_output) > 1:
                        raise ValueError('axes_inserting_output cannot contain repeated values')

            except TypeError:
                if axes_inserting_output is None:
                    axes_inserting_output = np.arange(n_axes_inserting_output)
                else:
                    if axes_inserting_output < 0:
                        axes_inserting_output += n_axes_array_output_object
                    axes_inserting_output = np.asarray([axes_inserting_output], dtype=int)

            if n_axes_other_array_input == 0:
                if dtype is not None and dtype != array_data_staying.dtype:
                    array_data_staying = array_data_staying.astype(dtype)
            else:
                if dtype is None:
                    dtype = array_data_staying.dtype
        else:
            n_axes_inserting_output = axes_inserting_output.size
            n_axes_array_output_object = n_axes_inserting_output

        indexes_array_data_staying = np.empty(n_axes_array_input, dtype=object)
        indexes_array_data_staying[:] = slice(None)

        axis_variables_in_combinations = int(axis_indexes_adding_axes_input > axis_spitting_input)
        axis_combinations_in_combinations = int(not (bool(axis_variables_in_combinations)))

        if n_axes_other_array_input == 0:
            conditions_indexes_table_adding_axes = trials_to_conditions(
                array_indexes_adding_axes, axis_combinations=axis_combinations_in_combinations)
        elif n_axes_other_array_input > 0:
            # ---------------------------------------------------------------------------------------------
            # FAST BUT MAYBE WRONG IN SOME CASES 
            indexes_array_indexes_adding_axes = np.copy(indexes_array_data_staying)
            # indexes_array_indexes_adding_axes[axes_other_array_input] = 0
            # array_indexes_adding_axes_1 = array_indexes_adding_axes[tuple(indexes_array_indexes_adding_axes)]
            # conditions_indexes_table_adding_axes = trials_to_conditions(
            #     array_indexes_adding_axes_1, axis_combinations=axis_combinations_in_combinations)
            # ----------------------------------------------------------------------------------------
            
            # ---------------------------------------------------------------------------------------
            # SLOW BUT SAFIER IN SOME CASES
            array_indexes_adding_axes_flat = merge_axes_and_add_axes_removing_info_to_axis_receiving(
                array_indexes_adding_axes,
                axes_other_array_input,
                axis_spitting_input,
                axis_indexes_adding_axes_input,
                indexes_inserting_output=np.arange(n_axes_other_array_input),
                dtype=array_indexes_adding_axes.dtype)

            indexes = np.empty(2, dtype=object)
            indexes[axis_combinations_in_combinations] = slice(0, None, 1)
            indexes[axis_variables_in_combinations] = slice(n_axes_other_array_input, None, 1)
            array_indexes_removing_object = array_indexes_adding_axes_flat[tuple(indexes)]
            # indexes[axis_variables_in_combinations] = slice(0, n_axes_other_array_input, 1)
            # array_indexes_removing_not_object = array_indexes_adding_axes_flat[tuple(indexes)]
            conditions_indexes_table_adding_axes = trials_to_conditions(
                array_indexes_removing_object, axis_combinations=axis_combinations_in_combinations)

            # conditions_indexes_removing_not_object = trials_to_conditions(
            #     array_indexes_removing_not_object, axis_combinations=axis_combinations_in_combinations)
            # ----------------------------------------------------------------------------------------

        n_conditions_indexes_table_adding_axes = conditions_to_n_conditions(
            conditions_indexes_table_adding_axes)
        n_combinations_indexes_table_adding_axes = int(np.prod(
            n_conditions_indexes_table_adding_axes))

        combinations_indexes_table_adding_axes = conditions_to_combinations(
            conditions_indexes_table_adding_axes,
            axis_combinations=axis_combinations_in_combinations)

        combinations_axes_inserting_output = n_conditions_to_combinations(
            n_conditions_indexes_table_adding_axes)

        # axes_array_output_object = np.arange(n_axes_array_output_object)
        shape_array_output_object = np.empty(n_axes_array_output_object, dtype=int)
        shape_array_output_object[axes_inserting_output] = n_conditions_indexes_table_adding_axes
        array_output = np.empty(shape_array_output_object, dtype=object)

        indexes_output_object = np.empty(n_axes_array_output_object, dtype=object)

        indexes_combinations = np.empty(2, dtype=object)
        indexes_combinations[axis_variables_in_combinations] = slice(None)

        if n_axes_other_array_input == 0:

            for i in range(n_combinations_indexes_table_adding_axes):
                indexes_output_object[axes_inserting_output] = combinations_axes_inserting_output[i]
                indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                indexes_array_data_staying[axis_spitting_input] = np.all(
                    array_indexes_adding_axes ==
                    combinations_indexes_table_adding_axes[tuple(indexes_combinations)],
                    axis=axis_variables_in_combinations)
                array_output[tuple(indexes_output_object)] = array_data_staying[tuple(
                    indexes_array_data_staying)]

        elif n_axes_other_array_input > 0:

            shape_array_output = np.copy(shape_array_data_staying)

            indexes_output = np.empty(n_axes_array_input, dtype=object)
            indexes_output[:] = slice(None)
            indexes_array_data_staying_j = np.empty(2, dtype=object)
            indexes_array_data_staying_j[axis_variables_in_combinations] = slice(None)

            combinations_axes_other_0 = np.empty(n_axes_other_array_input, dtype=int)
            combinations_axes_other_0[:] = 0
            indexes_output[axes_other_array_input] = combinations_axes_other_0
            indexes_output_tuple = tuple(indexes_output)
            indexes_array_data_staying[axes_other_array_input] = combinations_axes_other_0
            indexes_array_indexes_adding_axes[axes_other_array_input] = combinations_axes_other_0
            array_indexes_adding_axes_0 = array_indexes_adding_axes[
                tuple(indexes_array_indexes_adding_axes)]
            array_data_staying_0 = array_data_staying[tuple(
                indexes_array_data_staying)]
            for i in range(n_combinations_indexes_table_adding_axes):
                indexes_output_object[axes_inserting_output] = combinations_axes_inserting_output[i]
                indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                indexes_array_data_staying_j[axis_combinations_in_combinations] = np.all(
                    array_indexes_adding_axes_0 ==
                    combinations_indexes_table_adding_axes[tuple(indexes_combinations)],
                    axis=axis_variables_in_combinations)

                shape_array_output[axis_spitting_input] = np.sum(
                    indexes_array_data_staying_j[axis_combinations_in_combinations])

                array_output[tuple(indexes_output_object)] = np.empty(shape_array_output, dtype=dtype)

                array_output[tuple(indexes_output_object)][indexes_output_tuple] = array_data_staying_0[
                    tuple(indexes_array_data_staying_j)]

            combinations_axes_other = n_conditions_to_combinations(
                shape_array_data_staying[axes_other_array_input])
            n_combinations_axes_other = combinations_axes_other.shape[0]

            for j in range(1, n_combinations_axes_other):
                combinations_axes_other_j = combinations_axes_other[j]
                indexes_output[axes_other_array_input] = combinations_axes_other_j
                indexes_output_tuple = tuple(indexes_output)
                indexes_array_data_staying[axes_other_array_input] = combinations_axes_other_j
                indexes_array_indexes_adding_axes[axes_other_array_input] = combinations_axes_other_j
                array_indexes_adding_axes_j = array_indexes_adding_axes[
                    tuple(indexes_array_indexes_adding_axes)]
                array_data_staying_j = array_data_staying[tuple(
                    indexes_array_data_staying)]
                for i in range(n_combinations_indexes_table_adding_axes):
                    indexes_output_object[axes_inserting_output] = combinations_axes_inserting_output[i]
                    indexes_combinations[axis_combinations_in_combinations] = slice(i, i + 1)
                    indexes_array_data_staying_j[axis_combinations_in_combinations] = np.all(
                        array_indexes_adding_axes_j ==
                        combinations_indexes_table_adding_axes[tuple(indexes_combinations)],
                        axis=axis_variables_in_combinations)
                    array_output[tuple(indexes_output_object)][indexes_output_tuple] = array_data_staying_j[
                        tuple(indexes_array_data_staying_j)]

    return array_output


def from_2_arrays_advanced(
        array_data_staying,
        array_indexes_adding_axes,
        axis_spitting_input,
        axis_indexes_adding_axes_input,
        indexes_table_adding_axes=None,
        axes_inserting_output=None,
        indexes_staying=None,
        balance=True,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) uses the efficient function slice for indexes;
    # 2) it splits the array in two arrays first: array_indexes_adding_axes and array_data_staying.
    #    Then, it makes the array output from those two arrays;
    # 3) it assumes that the numbers of samples (or trials) in the axis_spitting_input for all combinations
    #    of indexes_table_adding_axes are equal;
    # 4) it does not assumes that the order of samples (or trials) in the axis_spitting_input are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_indexes_adding_axes_input. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) the numbers of axes of the two arrays must be equal;
    # 2) the shapes of the 2 arrays must be equal, except for the axis_variable_table,
    #    i.e. the axis that contains the variables of the tables;
    # 3) axes_inserting_output cannot contain repeated values;
    # 4) indexes_table_adding_axes cannot contain repeated values;
    # 5) shapes of axes_inserting_output and indexes_table_adding_axes must be equal
    # 6) axis_spitting_input != axis_indexes_adding_axes_input
    # 7) the numbers of samples (or trials) in the axis_spitting_input for all combinations
    #    of indexes_table_adding_axes are equal

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

    shape_array_data_staying = np.asarray(array_data_staying.shape)
    n_axes_array_data_staying = shape_array_data_staying.size
    n_variables_table_in_array_data_staying = shape_array_data_staying[axis_indexes_adding_axes_input]
    variables_table_in_array_data_staying = np.arange(n_variables_table_in_array_data_staying)

    shape_array_indexes_adding_axes = np.asarray(array_indexes_adding_axes.shape)
    n_axes_array_indexes_adding_axes = shape_array_indexes_adding_axes.size
    n_variables_table_in_array_indexes_adding_axes = shape_array_indexes_adding_axes[axis_indexes_adding_axes_input]
    variables_table_in_array_indexes_adding_axes = np.arange(n_variables_table_in_array_indexes_adding_axes)

    if format_and_check:
        # check point 1
        if n_axes_array_data_staying != n_axes_array_indexes_adding_axes:
            raise ValueError('dimension mismatch')
        else:
            n_axes_array_input = n_axes_array_data_staying

        if axis_indexes_adding_axes_input < 0:
            axis_indexes_adding_axes_input += n_axes_array_input
        if axis_spitting_input < 0:
            axis_spitting_input += n_axes_array_input

        # check point 6
        if axis_spitting_input == axis_indexes_adding_axes_input:
            raise ValueError('axis_spitting_input and axis_indexes_adding_axes_input must be different')

        # check point 2
        indexes_logical = np.arange(n_axes_array_input) != axis_indexes_adding_axes_input
        if np.any(shape_array_data_staying[indexes_logical] != shape_array_indexes_adding_axes[indexes_logical]):
            raise ValueError('dimension mismatch')

        n_variables_table_in_array_data_staying = shape_array_data_staying[axis_indexes_adding_axes_input]
        variables_table_in_array_data_staying = np.arange(n_variables_table_in_array_data_staying)

        n_variables_table_in_array_indexes_adding_axes = shape_array_indexes_adding_axes[axis_indexes_adding_axes_input]
        variables_table_in_array_indexes_adding_axes = np.arange(n_variables_table_in_array_indexes_adding_axes)

        # format indexes_table_adding_axes
        try:
            n_indexes_table_adding_axes = len(indexes_table_adding_axes)
            indexes_table_adding_axes = np.asarray(indexes_table_adding_axes, dtype=int)
            indexes_table_adding_axes[indexes_table_adding_axes < 0] += \
                n_variables_table_in_array_indexes_adding_axes
            # check point 4
            for v in indexes_table_adding_axes:
                if np.sum(v == indexes_table_adding_axes) > 1:
                    raise ValueError('indexes_table_adding_axes cannot contain repeated values')
        except TypeError:
            if indexes_table_adding_axes is None:
                indexes_table_adding_axes = np.arange(n_variables_table_in_array_indexes_adding_axes)
                n_indexes_table_adding_axes = n_variables_table_in_array_indexes_adding_axes
            else:
                if indexes_table_adding_axes < 0:
                    indexes_table_adding_axes += n_variables_table_in_array_indexes_adding_axes
                indexes_table_adding_axes = np.asarray([indexes_table_adding_axes], dtype=int)
                n_indexes_table_adding_axes = 1

        # format indexes_staying
        try:
            n_indexes_staying = len(indexes_staying)
            indexes_staying = np.asarray(indexes_staying, dtype=int)
            indexes_staying[indexes_staying < 0] += n_variables_table_in_array_data_staying
        except TypeError:
            if indexes_staying is None:
                indexes_staying = np.arange(n_variables_table_in_array_data_staying)
                n_indexes_staying = n_variables_table_in_array_data_staying
            else:
                if indexes_staying < 0:
                    indexes_staying += n_variables_table_in_array_data_staying
                indexes_staying = np.asarray([indexes_staying], dtype=int)
                n_indexes_staying = 1

        # format axes_inserting_output
        n_axes_inserting_output = n_indexes_table_adding_axes
        if balance:
            n_axes_array_output = n_axes_array_input + n_axes_inserting_output
        else:
            n_axes_array_output = n_axes_inserting_output
        try:
            n_axes_inserting_output = len(axes_inserting_output)
            axes_inserting_output = np.asarray(axes_inserting_output, dtype=int)
            axes_inserting_output[axes_inserting_output < 0] += n_axes_array_output
            # check point 3
            for a in axes_inserting_output:
                if np.sum(a == axes_inserting_output) > 1:
                    raise ValueError('axes_inserting_output cannot contain repeated values')
            # check point 5
            if n_indexes_table_adding_axes != n_axes_inserting_output:
                raise ValueError('Shapes of axes_inserting_output and indexes_table_adding_axes must be equal')
        except TypeError:
            if axes_inserting_output is None:
                axes_inserting_output = np.arange(n_axes_inserting_output)
            else:
                if axes_inserting_output < 0:
                    axes_inserting_output += n_axes_array_output
                axes_inserting_output = np.asarray([axes_inserting_output], dtype=int)
                n_axes_inserting_output = 1
                # check point 5
                if n_indexes_table_adding_axes != n_axes_inserting_output:
                    raise ValueError('Shapes of axes_inserting_output and indexes_table_adding_axes must be equal')
    else:
        n_indexes_staying = indexes_staying.size
        n_indexes_table_adding_axes = indexes_table_adding_axes.size

    indexes = None
    if ((n_indexes_staying != n_variables_table_in_array_data_staying) or
            np.any(indexes_staying != variables_table_in_array_data_staying)):

        indexes = np.empty(n_axes_array_data_staying, dtype=object)
        indexes[:] = slice(None)
        indexes[axis_indexes_adding_axes_input] = indexes_staying
        array_data_staying = array_data_staying[tuple(indexes)]

    if ((n_indexes_table_adding_axes != n_variables_table_in_array_indexes_adding_axes) or
            np.any(indexes_table_adding_axes != variables_table_in_array_indexes_adding_axes)):

        if indexes is None:
            indexes = np.empty(n_axes_array_indexes_adding_axes, dtype=object)
            indexes[:] = slice(None)
        indexes[axis_indexes_adding_axes_input] = indexes_table_adding_axes
        array_indexes_adding_axes = array_indexes_adding_axes[tuple(indexes)]

    array_output = from_2_arrays(
        array_data_staying,
        array_indexes_adding_axes,
        axis_spitting_input,
        axis_indexes_adding_axes_input,
        axes_inserting_output=axes_inserting_output,
        balance=balance,
        dtype=dtype,
        format_and_check=False)

    return array_output


def from_1_array(
        array_input,
        axis_spitting_input,
        axis_indexes_adding_axes_input,
        indexes_table_adding_axes,
        axes_inserting_output=None,
        indexes_staying=None,
        balance=True,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) uses the efficient function slice for indexes;
    # 2) it splits the array in two arrays first: array_indexes_adding_axes and array_data_staying.
    #    Then, it makes the array output from those two arrays;
    # 3) the numbers of samples (or trials) in the axis_spitting_input for all combinations
    #    of indexes_table_adding_axes have to be equal;
    # 4) it does not assumes that the order of samples (or trials) in the axis_spitting_input are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_indexes_adding_axes_input. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) axes_inserting_output cannot contain repeated values;
    # 2) indexes_table_adding_axes cannot contain repeated values;
    # 3) shapes of axes_inserting_output and indexes_table_adding_axes must be equal
    # 4) axis_spitting_input != axis_indexes_adding_axes_input
    # 5) the numbers of samples (or trials) in the axis_spitting_input for all combinations
    #    of indexes_table_adding_axes are equal

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

    shape_array_input = np.asarray(array_input.shape)
    n_axes_array_input = shape_array_input.size
    n_variables_table_input = shape_array_input[axis_indexes_adding_axes_input]
    variables_table_input = np.arange(n_variables_table_input)

    if format_and_check:
        if axis_indexes_adding_axes_input < 0:
            axis_indexes_adding_axes_input += n_axes_array_input
        if axis_spitting_input < 0:
            axis_spitting_input += n_axes_array_input

        # check point 4
        if axis_spitting_input == axis_indexes_adding_axes_input:
            raise ValueError('axis_spitting_input and axis_indexes_adding_axes_input must be different')

        # format indexes_table_adding_axes
        try:
            n_indexes_table_adding_axes = len(indexes_table_adding_axes)
            indexes_table_adding_axes = np.asarray(indexes_table_adding_axes, dtype=int)
            indexes_table_adding_axes[indexes_table_adding_axes < 0] += n_variables_table_input
            # check point 2
            for v in indexes_table_adding_axes:
                if np.sum(v == indexes_table_adding_axes) > 1:
                    raise ValueError('indexes_table_adding_axes cannot contain repeated values')
        except TypeError:
            if indexes_table_adding_axes < 0:
                indexes_table_adding_axes += n_variables_table_input
            indexes_table_adding_axes = np.asarray([indexes_table_adding_axes], dtype=int)
            n_indexes_table_adding_axes = 1

        # format indexes_staying
        try:
            n_indexes_staying = len(indexes_staying)
            indexes_staying = np.asarray(indexes_staying, dtype=int)
            indexes_staying[indexes_staying < 0] += n_variables_table_input
        except TypeError:
            if indexes_staying is None:
                indexes_staying = variables_table_input[np.logical_not(
                    samples_in_arr1_are_in_arr2(variables_table_input, indexes_table_adding_axes))]
                n_indexes_staying = indexes_staying.size
            else:
                if indexes_staying < 0:
                    indexes_staying += n_variables_table_input
                indexes_staying = np.asarray([indexes_staying], dtype=int)
                n_indexes_staying = 1

        # format axes_inserting_output
        n_axes_inserting_output = n_indexes_table_adding_axes
        if balance:
            n_axes_array_output = n_axes_array_input + n_axes_inserting_output
        else:
            n_axes_array_output = n_axes_inserting_output
        try:
            n_axes_inserting_output = len(axes_inserting_output)
            axes_inserting_output = np.asarray(axes_inserting_output, dtype=int)
            axes_inserting_output[axes_inserting_output < 0] += n_axes_array_output
            # check point 1
            for a in axes_inserting_output:
                if np.sum(a == axes_inserting_output) > 1:
                    raise ValueError('axes_inserting_output cannot contain repeated values')
            # check point 3
            if n_indexes_table_adding_axes != n_axes_inserting_output:
                raise ValueError('Shapes of axes_inserting_output and indexes_table_adding_axes must be equal')
        except TypeError:
            if axes_inserting_output is None:
                axes_inserting_output = np.arange(n_axes_inserting_output)
            else:
                if axes_inserting_output < 0:
                    axes_inserting_output += n_axes_array_output
                axes_inserting_output = np.asarray([axes_inserting_output], dtype=int)
                n_axes_inserting_output = 1
                # check point 3
                if n_indexes_table_adding_axes != n_axes_inserting_output:
                    raise ValueError('Shapes of axes_inserting_output and indexes_table_adding_axes must be equal')
    else:
        n_indexes_staying = indexes_staying.size
        n_indexes_table_adding_axes = indexes_table_adding_axes.size

    indexes = np.empty(n_axes_array_input, dtype=object)
    indexes[:] = slice(None)
    if ((n_indexes_staying != n_variables_table_input) or
            np.any(indexes_staying != variables_table_input)):
        indexes[axis_indexes_adding_axes_input] = indexes_staying
    array_data_staying = array_input[tuple(indexes)]

    if ((n_indexes_table_adding_axes != n_variables_table_input) or
            np.any(indexes_table_adding_axes != variables_table_input)):
        indexes[axis_indexes_adding_axes_input] = indexes_table_adding_axes
    else:
        indexes[axis_indexes_adding_axes_input] = slice(None)
    array_indexes_adding_axes = array_input[tuple(indexes)]
    del array_input

    array_output = from_2_arrays(
        array_data_staying,
        array_indexes_adding_axes,
        axis_spitting_input,
        axis_indexes_adding_axes_input,
        axes_inserting_output=axes_inserting_output,
        balance=balance,
        dtype=dtype,
        format_and_check=False)

    return array_output
