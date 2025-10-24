import numpy as np
from ...array import samples_in_arr1_are_in_arr2
from .balanced_or_unbalanced import from_2_arrays as v2a_from_2_arrays
# from .balanced import from_2_arrays as v2a_balanced_from_2_arrays
# from .unbalanced import from_2_arrays as v2a_unbalanced_from_2_arrays


def from_3_arrays(
        array_variables_staying,
        array_variables_adding_axes_balanced,
        array_variables_adding_axes_unbalanced,
        axis_samples,
        axis_variables_table,
        axes_inserting_array_not_object=None,
        axes_inserting_array_object=None,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) it is safe;
    # 2) it does not assumes that the numbers of samples (or trials) in the axis_samples for all
    #    combinations of variables_table_adding_axes_balanced are equal;
    # 3) it does not assumes that the order of samples (or trials) in the axis_samples are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_variables_table. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) axes_inserting_array_not_object cannot contain repeated values;
    # 2) axes_inserting_array_object cannot contain repeated values;
    # 3) axis_samples != axis_variables_table
    # 4) the numbers of axes of the 3 arrays must be equal;
    # 5) the shapes of the 3 arrays must be equal, except for the axis_variable_table,
    #     i.e. the axis that contains the variables of the tables;
    # 6) Either one of the 3 conditions must be True:
    #    array_variables_adding_axes_balanced != None;
    #    array_variables_adding_axes_unbalanced != None;
    #    array_variables_adding_axes_balanced != None and array_variables_adding_axes_unbalanced != None;

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

    # check point 6
    if array_variables_adding_axes_balanced is None and array_variables_adding_axes_unbalanced is None:
        raise ValueError('Either one of the 3 conditions must be True:'
                         'array_variables_adding_axes_balanced != None\n'
                         'array_variables_adding_axes_unbalanced != None\n'
                         'array_variables_adding_axes_balanced != None and '
                         'array_variables_adding_axes_unbalanced != None;')

    axis_variables_table_input = axis_variables_table
    axis_samples_input = axis_samples

    if format_and_check:

        shape_array_variables_staying = np.asarray(array_variables_staying.shape)
        shape_array_variables_adding_axes_balanced = np.asarray(array_variables_adding_axes_balanced.shape)
        shape_array_variables_adding_axes_unbalanced = np.asarray(array_variables_adding_axes_unbalanced.shape)

        # check point 4
        n_axes_array_variables_staying = shape_array_variables_staying.size
        n_axes_array_variables_adding_axes_balanced = shape_array_variables_adding_axes_balanced.size
        n_axes_array_variables_adding_axes_unbalanced = shape_array_variables_adding_axes_unbalanced.size
        if (n_axes_array_variables_staying != n_axes_array_variables_adding_axes_balanced) or (
                n_axes_array_variables_staying != n_axes_array_variables_adding_axes_unbalanced):
            raise ValueError('dimension mismatch')
        else:
            n_axes_array_input = n_axes_array_variables_staying

        if axis_variables_table_input < 0:
            axis_variables_table_input += n_axes_array_input
        if axis_samples_input < 0:
            axis_samples_input += n_axes_array_input

        # check point 3
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        # check point 5
        indexes_logical = np.arange(n_axes_array_input) != axis_variables_table_input
        if np.any(shape_array_variables_staying[indexes_logical] !=
                  shape_array_variables_adding_axes_balanced[indexes_logical]):
            raise ValueError('dimension mismatch')
        if np.any(shape_array_variables_staying[indexes_logical] !=
                  shape_array_variables_adding_axes_unbalanced[indexes_logical]):
            raise ValueError('dimension mismatch')

        if array_variables_adding_axes_balanced is not None:
            # format axes_inserting_array_not_object
            n_variables_table_adding_axes_balanced = (
                shape_array_variables_adding_axes_balanced[axis_variables_table_input])
            n_axes_inserting_array_not_object = n_variables_table_adding_axes_balanced
            n_axes_array_not_object = n_axes_array_variables_staying + n_axes_inserting_array_not_object
            try:
                len(axes_inserting_array_not_object)
                axes_inserting_array_not_object = np.asarray(axes_inserting_array_not_object, dtype=int)
                axes_inserting_array_not_object[axes_inserting_array_not_object < 0] += n_axes_array_not_object
                # check point 1
                for a in axes_inserting_array_not_object:
                    if np.sum(a == axes_inserting_array_not_object) > 1:
                        raise ValueError('axes_inserting_array_not_object cannot contain repeated values')
            except TypeError:
                if axes_inserting_array_not_object is None:
                    axes_inserting_array_not_object = np.arange(n_axes_inserting_array_not_object)
                else:
                    if axes_inserting_array_not_object < 0:
                        axes_inserting_array_not_object += n_axes_array_not_object
                    axes_inserting_array_not_object = np.asarray([axes_inserting_array_not_object], dtype=int)

        if array_variables_adding_axes_unbalanced is not None:
            # format axes_inserting_array_object
            n_variables_table_adding_axes_unbalanced = (
                shape_array_variables_adding_axes_balanced[axis_variables_table_input])
            n_axes_inserting_array_object = n_variables_table_adding_axes_unbalanced
            n_axes_array_object = n_axes_inserting_array_object
            try:
                len(axes_inserting_array_object)
                axes_inserting_array_object = np.asarray(axes_inserting_array_object, dtype=int)
                axes_inserting_array_object[axes_inserting_array_object < 0] += n_axes_array_object
                # check point 2
                for a in axes_inserting_array_object[0]:
                    if np.sum(a == axes_inserting_array_object) > 1:
                        raise ValueError('axes_inserting_array_object cannot contain repeated values')
            except TypeError:
                if axes_inserting_array_object is None:
                    axes_inserting_array_object = np.arange(n_axes_inserting_array_object)
                else:
                    if axes_inserting_array_object < 0:
                        axes_inserting_array_object += n_axes_array_object
                    axes_inserting_array_object = np.asarray([axes_inserting_array_object], dtype=int)

    if array_variables_adding_axes_balanced is not None and array_variables_adding_axes_unbalanced is not None:

        array_variables_staying = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

        array_variables_adding_axes_unbalanced = v2a_from_2_arrays(
            array_variables_adding_axes_unbalanced,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=None,
            format_and_check=False)

        axis_variables_table_input += np.sum(axes_inserting_array_not_object <= axis_variables_table_input)
        axis_samples_input += np.sum(axes_inserting_array_not_object <= axis_samples_input)

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    elif array_variables_adding_axes_balanced is not None and array_variables_adding_axes_unbalanced is None:

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

    elif array_variables_adding_axes_balanced is None and array_variables_adding_axes_unbalanced is not None:

        if dtype is not None:
            array_variables_staying = array_variables_staying.astype(dtype)

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    return array_output


def from_3_arrays_advanced(
        array_variables_staying,
        array_variables_adding_axes_balanced,
        array_variables_adding_axes_unbalanced,
        axis_samples,
        axis_variables_table,
        variables_table_adding_axes_balanced=None,
        variables_table_adding_axes_unbalanced=None,
        axes_inserting_array_not_object=None,
        axes_inserting_array_object=None,
        variables_table_staying=None,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) it is safe;
    # 2) it does not assumes that the numbers of samples (or trials) in the axis_samples for all
    #    combinations of variables_table_adding_axes_balanced are equal;
    # 3) it does not assumes that the order of samples (or trials) in the axis_samples are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_variables_table. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) axes_inserting_array_not_object cannot contain repeated values;
    # 2) axes_inserting_array_object cannot contain repeated values;
    # 3) variables_table_adding_axes_balanced cannot contain repeated values;
    # 4) variables_table_adding_axes_unbalanced cannot contain repeated values;
    # 5) shapes of axes_inserting_array_not_object and variables_table_adding_axes_balanced must be equal
    # 6) shapes of axes_inserting_array_object and variables_table_adding_axes_unbalanced must be equal
    # 7) axis_samples != axis_variables_table
    # 8) Any value in variables_table_adding_axes_balanced must be different to
    #    any value in variables_table_adding_axes_unbalanced;
    # 9) Either one of the 3 conditions must be True:
    #    variables_table_adding_axes_balanced != None;
    #    variables_table_adding_axes_unbalanced != None
    #    variables_table_adding_axes_balanced != None and variables_table_adding_axes_unbalanced != None;
    # 10) the numbers of axes of the 3 arrays must be equal;
    # 11) the shapes of the 3 arrays must be equal, except for the axis_variable_table,
    #     i.e. the axis that contains the variables of the tables;

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

    # check point 9
    if variables_table_adding_axes_balanced is None and variables_table_adding_axes_unbalanced is None:
        raise ValueError('Either one of the 3 conditions must be True:'
                         'variables_table_adding_axes_balanced != None\n'
                         'variables_table_adding_axes_unbalanced != None\n'
                         'variables_table_adding_axes_balanced != None and '
                         'variables_table_adding_axes_unbalanced != None;')

    axis_variables_table_input = axis_variables_table
    axis_samples_input = axis_samples

    shape_array_variables_staying = np.asarray(array_variables_staying.shape)
    n_axes_array_variables_staying = shape_array_variables_staying.size
    n_variables_table_in_array_variables_staying = shape_array_variables_staying[axis_variables_table_input]
    variables_table_in_array_variables_staying = np.arange(n_variables_table_in_array_variables_staying)

    shape_array_variables_adding_axes_balanced = np.asarray(array_variables_adding_axes_balanced.shape)
    n_axes_array_variables_adding_axes_balanced = shape_array_variables_adding_axes_balanced.size
    n_variables_table_in_array_variables_adding_axes_balanced = (
        shape_array_variables_adding_axes_balanced[axis_variables_table_input])
    variables_table_in_array_variables_adding_axes_balanced = np.arange(
        n_variables_table_in_array_variables_adding_axes_balanced)

    shape_array_variables_adding_axes_unbalanced = np.asarray(array_variables_adding_axes_unbalanced.shape)
    n_axes_array_variables_adding_axes_unbalanced = shape_array_variables_adding_axes_unbalanced.size
    n_variables_table_in_array_variables_adding_axes_unbalanced = (
        shape_array_variables_adding_axes_unbalanced[axis_variables_table_input])
    variables_table_in_array_variables_adding_axes_unbalanced = np.arange(
        n_variables_table_in_array_variables_adding_axes_unbalanced)

    if format_and_check:
        # check point 10
        if (n_axes_array_variables_staying != n_axes_array_variables_adding_axes_balanced) or (
                n_axes_array_variables_staying != n_axes_array_variables_adding_axes_unbalanced):
            raise ValueError('dimension mismatch')
        else:
            n_axes_array_input = n_axes_array_variables_staying

        if axis_variables_table_input < 0:
            axis_variables_table_input += n_axes_array_input
        if axis_samples_input < 0:
            axis_samples_input += n_axes_array_input

        # check point 7
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        # check point 11
        indexes_logical = np.arange(n_axes_array_input) != axis_variables_table_input
        if np.any(shape_array_variables_staying[indexes_logical] !=
                  shape_array_variables_adding_axes_balanced[indexes_logical]):
            raise ValueError('dimension mismatch')
        if np.any(shape_array_variables_staying[indexes_logical] !=
                  shape_array_variables_adding_axes_unbalanced[indexes_logical]):
            raise ValueError('dimension mismatch')

        # format variables_table_staying
        try:
            n_variables_table_staying = len(variables_table_staying)
            variables_table_staying = np.asarray(variables_table_staying, dtype=int)
            variables_table_staying[variables_table_staying < 0] += n_variables_table_in_array_variables_staying
        except TypeError:
            if variables_table_staying is None:
                variables_table_staying = variables_table_in_array_variables_staying
                n_variables_table_staying = n_variables_table_in_array_variables_staying
            else:
                if variables_table_staying < 0:
                    variables_table_staying += n_variables_table_in_array_variables_staying
                variables_table_staying = np.asarray([variables_table_staying], dtype=int)
                n_variables_table_staying = 1

        if variables_table_adding_axes_balanced is not None:
            # format variables_table_adding_axes_balanced
            try:
                n_variables_table_adding_axes_balanced = len(variables_table_adding_axes_balanced)
                variables_table_adding_axes_balanced = np.asarray(variables_table_adding_axes_balanced, dtype=int)
                variables_table_adding_axes_balanced[variables_table_adding_axes_balanced < 0] += (
                    n_variables_table_in_array_variables_adding_axes_balanced)
                # check point 3
                for v in variables_table_adding_axes_balanced:
                    if np.sum(v == variables_table_adding_axes_balanced) > 1:
                        raise ValueError('variables_table_adding_axes_balanced cannot contain repeated values')
            except TypeError:
                if variables_table_adding_axes_balanced < 0:
                    variables_table_adding_axes_balanced += n_variables_table_in_array_variables_adding_axes_balanced
                variables_table_adding_axes_balanced = np.asarray([variables_table_adding_axes_balanced], dtype=int)
                n_variables_table_adding_axes_balanced = 1

            # format axes_inserting_array_not_object
            n_axes_inserting_array_not_object = n_variables_table_adding_axes_balanced
            n_axes_array_not_object = n_axes_array_variables_staying + n_axes_inserting_array_not_object
            try:
                n_axes_inserting_array_not_object = len(axes_inserting_array_not_object)
                axes_inserting_array_not_object = np.asarray(axes_inserting_array_not_object, dtype=int)
                axes_inserting_array_not_object[axes_inserting_array_not_object < 0] += n_axes_array_not_object
                # check point 1
                for a in axes_inserting_array_not_object:
                    if np.sum(a == axes_inserting_array_not_object) > 1:
                        raise ValueError('axes_inserting_array_not_object cannot contain repeated values')
                # check point 5
                if n_variables_table_adding_axes_balanced != n_axes_inserting_array_not_object:
                    raise ValueError('Shapes of axes_inserting_array_not_object and '
                                     'variables_table_adding_axes_balanced must be equal')
            except TypeError:
                if axes_inserting_array_not_object is None:
                    axes_inserting_array_not_object = np.arange(n_axes_inserting_array_not_object)
                else:
                    if axes_inserting_array_not_object < 0:
                        axes_inserting_array_not_object += n_axes_array_not_object
                    axes_inserting_array_not_object = np.asarray([axes_inserting_array_not_object], dtype=int)
                    n_axes_inserting_array_not_object = 1
                    # check point 5
                    if n_variables_table_adding_axes_balanced != n_axes_inserting_array_not_object:
                        raise ValueError('Shapes of axes_inserting_array_not_object and '
                                         'variables_table_adding_axes_balanced must be equal')

        if variables_table_adding_axes_unbalanced is not None:
            # format variables_table_adding_axes_unbalanced
            try:
                n_variables_table_adding_axes_unbalanced = len(variables_table_adding_axes_unbalanced)
                variables_table_adding_axes_unbalanced = np.asarray(variables_table_adding_axes_unbalanced, dtype=int)
                variables_table_adding_axes_unbalanced[variables_table_adding_axes_unbalanced < 0] += (
                    n_variables_table_in_array_variables_adding_axes_balanced)
                # check point 4
                for v in variables_table_adding_axes_unbalanced:
                    if np.sum(v == variables_table_adding_axes_unbalanced) > 1:
                        raise ValueError('variables_table_adding_axes_unbalanced cannot contain repeated values')
            except TypeError:
                if variables_table_adding_axes_unbalanced < 0:
                    variables_table_adding_axes_unbalanced += n_variables_table_in_array_variables_adding_axes_balanced
                variables_table_adding_axes_unbalanced = np.asarray([variables_table_adding_axes_unbalanced], dtype=int)
                n_variables_table_adding_axes_unbalanced = 1

            # format axes_inserting_array_object
            n_axes_inserting_array_object = n_variables_table_adding_axes_unbalanced
            n_axes_array_object = n_axes_inserting_array_object
            try:
                n_axes_inserting_array_object = len(axes_inserting_array_object)
                axes_inserting_array_object = np.asarray(axes_inserting_array_object, dtype=int)
                axes_inserting_array_object[axes_inserting_array_object < 0] += n_axes_array_object
                # check point 2
                for a in axes_inserting_array_object:
                    if np.sum(a == axes_inserting_array_object) > 1:
                        raise ValueError('axes_inserting_array_object cannot contain repeated values')
                # check point 6
                if n_variables_table_adding_axes_unbalanced != n_axes_inserting_array_object:
                    raise ValueError('Shapes of axes_inserting_array_object and '
                                     'variables_table_adding_axes_unbalanced must be equal')
            except TypeError:
                if axes_inserting_array_object is None:
                    axes_inserting_array_object = np.arange(n_axes_inserting_array_object)
                else:
                    if axes_inserting_array_object < 0:
                        axes_inserting_array_object += n_axes_array_object
                    axes_inserting_array_object = np.asarray([axes_inserting_array_object], dtype=int)
                    n_axes_inserting_array_object = 1
                    # check point 6
                    if n_variables_table_adding_axes_unbalanced != n_axes_inserting_array_object:
                        raise ValueError('Shapes of axes_inserting_array_object and '
                                         'variables_table_adding_axes_unbalanced must be equal')
    else:
        n_variables_table_staying = variables_table_staying.size
        if variables_table_adding_axes_balanced is not None:
            n_variables_table_adding_axes_balanced = variables_table_adding_axes_balanced.size
        if variables_table_adding_axes_unbalanced is not None:
            n_variables_table_adding_axes_unbalanced = variables_table_adding_axes_unbalanced.size

    indexes = None
    if ((n_variables_table_staying != n_variables_table_in_array_variables_staying) or
            np.any(variables_table_staying != variables_table_in_array_variables_staying)):

        indexes = np.empty(n_axes_array_variables_staying, dtype=object)
        indexes[:] = slice(None)
        indexes[axis_variables_table_input] = variables_table_staying
        array_variables_staying = array_variables_staying[tuple(indexes)]

    if variables_table_adding_axes_balanced is not None:

        if ((n_variables_table_adding_axes_balanced != n_variables_table_in_array_variables_adding_axes_balanced)
                or np.any(variables_table_adding_axes_balanced !=
                          variables_table_in_array_variables_adding_axes_balanced)):

            if indexes is None:
                indexes = np.empty(n_axes_array_variables_adding_axes_balanced, dtype=object)
                indexes[:] = slice(None)
            indexes[axis_variables_table_input] = variables_table_adding_axes_balanced
            array_variables_adding_axes_balanced = array_variables_adding_axes_balanced[tuple(indexes)]

    if variables_table_adding_axes_unbalanced is not None:

        if ((n_variables_table_adding_axes_unbalanced != n_variables_table_in_array_variables_adding_axes_unbalanced)
                or np.any(variables_table_adding_axes_unbalanced !=
                          variables_table_in_array_variables_adding_axes_unbalanced)):
            if indexes is None:
                indexes = np.empty(n_axes_array_variables_adding_axes_unbalanced, dtype=object)
                indexes[:] = slice(None)
            indexes[axis_variables_table_input] = variables_table_adding_axes_unbalanced
            array_variables_adding_axes_unbalanced = array_variables_adding_axes_unbalanced[tuple(indexes)]

    if variables_table_adding_axes_balanced is not None and variables_table_adding_axes_unbalanced is not None:
        # from ccalafiore.array import samples_in_arr1_are_in_arr2
        # check point 8
        if np.any(samples_in_arr1_are_in_arr2(
                variables_table_adding_axes_balanced, variables_table_adding_axes_unbalanced)):
            raise ValueError('Any value in variables_table_adding_axes_balanced must be different to\n'
                             'any value in variables_table_adding_axes_unbalanced')

        array_variables_staying = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting_output=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

        array_variables_adding_axes_unbalanced = v2a_from_2_arrays(
            array_variables_adding_axes_unbalanced,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting_output=axes_inserting_array_not_object,
            balance=True,
            dtype=None,
            format_and_check=False)

        axis_variables_table_input += np.sum(axes_inserting_array_not_object <= axis_variables_table_input)
        axis_samples_input += np.sum(axes_inserting_array_not_object <= axis_samples_input)

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting_output=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    elif variables_table_adding_axes_balanced is not None and variables_table_adding_axes_unbalanced is None:

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

    elif variables_table_adding_axes_balanced is None and variables_table_adding_axes_unbalanced is not None:

        if dtype is not None:
            array_variables_staying = array_variables_staying.astype(dtype)

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    return array_output


def from_1_array(
        array_input,
        axis_samples,
        axis_variables_table,
        variables_table_adding_axes_balanced=None,
        variables_table_adding_axes_unbalanced=None,
        axes_inserting_array_not_object=None,
        axes_inserting_array_object=None,
        variables_table_staying=None,
        dtype=None,
        format_and_check=True):

    # Notes:
    # 1) it is safe;
    # 2) it does not assumes that the numbers of samples (or trials) in the axis_samples for all
    #    combinations of variables_table_adding_axes_balanced are equal;
    # 3) it does not assumes that the order of samples (or trials) in the axis_samples are is the same
    #    in each combination of variables' axes. The order is based on the order of the combinations
    #    of variables' conditions in axis_variables_table. In other words, the samples can be in random order
    #    (or in different order) in each combinations of variables axes/conditions.

    # Input requirements:
    # 1) axes_inserting_array_not_object cannot contain same values;
    # 2) axes_inserting_array_object cannot contain same values;
    # 3) variables_table_adding_axes_balanced cannot contain same values;
    # 4) variables_table_adding_axes_unbalanced cannot contain same values;
    # 5) shapes of axes_inserting_array_not_object and variables_table_adding_axes_balanced must be equal
    # 6) shapes of axes_inserting_array_object and variables_table_adding_axes_unbalanced must be equal
    # 7) axis_samples != axis_variables_table
    # 8) Any value in variables_table_adding_axes_balanced must be different to
    #    any value in variables_table_adding_axes_unbalanced;
    # 9) Either one of the 3 conditions must be True:
    #    variables_table_adding_axes_balanced != None;
    #    variables_table_adding_axes_unbalanced != None
    #    variables_table_adding_axes_balanced != None and variables_table_adding_axes_unbalanced != None;'

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

    # check point 9
    if variables_table_adding_axes_balanced is None and variables_table_adding_axes_unbalanced is None:
        raise ValueError('Either one of the 3 conditions must be True:'
                         'variables_table_adding_axes_balanced != None\n'
                         'variables_table_adding_axes_unbalanced != None\n'
                         'variables_table_adding_axes_balanced != None and '
                         'variables_table_adding_axes_unbalanced != None;')

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

        # check point 7
        if axis_samples_input == axis_variables_table_input:
            raise ValueError('axis_samples_input and axis_variables_table_input must be different')

        variables_table_adding_axes = None

        if variables_table_adding_axes_balanced is not None:

            # format variables_table_adding_axes_balanced
            try:
                n_variables_table_adding_axes_balanced = len(variables_table_adding_axes_balanced)
                variables_table_adding_axes_balanced = np.asarray(variables_table_adding_axes_balanced, dtype=int)
                variables_table_adding_axes_balanced[variables_table_adding_axes_balanced < 0] += (
                    n_variables_table_input)
                # check point 3
                for v in variables_table_adding_axes_balanced:
                    if np.sum(v == variables_table_adding_axes_balanced) > 1:
                        raise ValueError('variables_table_adding_axes_balanced cannot contain repeated values')
            except TypeError:
                if variables_table_adding_axes_balanced < 0:
                    variables_table_adding_axes_balanced += n_variables_table_input
                variables_table_adding_axes_balanced = np.asarray([variables_table_adding_axes_balanced], dtype=int)
                n_variables_table_adding_axes_balanced = 1

            variables_table_adding_axes = variables_table_adding_axes_unbalanced

            # format axes_inserting_array_not_object
            n_axes_inserting_array_not_object = n_variables_table_adding_axes_balanced
            n_axes_array_not_object = n_axes_array_input + n_axes_inserting_array_not_object
            try:
                n_axes_inserting_array_not_object = len(axes_inserting_array_not_object)
                axes_inserting_array_not_object = np.asarray(axes_inserting_array_not_object, dtype=int)
                axes_inserting_array_not_object[axes_inserting_array_not_object < 0] += n_axes_array_not_object
                # check point 1
                for a in axes_inserting_array_not_object:
                    if np.sum(a == axes_inserting_array_not_object) > 1:
                        raise ValueError('axes_inserting_array_not_object cannot contain repeated values')
                # check point 5
                if n_variables_table_adding_axes_balanced != n_axes_inserting_array_not_object:
                    raise ValueError('Shapes of axes_inserting_array_not_object and '
                                     'variables_table_adding_axes_balanced must be equal')
            except TypeError:
                if axes_inserting_array_not_object is None:
                    axes_inserting_array_not_object = np.arange(n_axes_inserting_array_not_object)
                else:
                    if axes_inserting_array_not_object < 0:
                        axes_inserting_array_not_object += n_axes_array_not_object
                    axes_inserting_array_not_object = np.asarray([axes_inserting_array_not_object], dtype=int)
                    n_axes_inserting_array_not_object = 1
                    # check point 5
                    if n_variables_table_adding_axes_balanced != n_axes_inserting_array_not_object:
                        raise ValueError('Shapes of axes_inserting_array_not_object and '
                                         'variables_table_adding_axes_balanced must be equal')

        if variables_table_adding_axes_unbalanced is not None:
            # format variables_table_adding_axes_unbalanced
            try:
                n_variables_table_adding_axes_unbalanced = len(variables_table_adding_axes_unbalanced)
                variables_table_adding_axes_unbalanced = np.asarray(variables_table_adding_axes_unbalanced, dtype=int)
                variables_table_adding_axes_unbalanced[variables_table_adding_axes_unbalanced < 0] += \
                    n_variables_table_input
                # check point 4
                for v in variables_table_adding_axes_unbalanced:
                    if np.sum(v == variables_table_adding_axes_unbalanced) > 1:
                        raise ValueError('variables_table_adding_axes_unbalanced cannot contain repeated values')
            except TypeError:
                if variables_table_adding_axes_unbalanced < 0:
                    variables_table_adding_axes_unbalanced += n_variables_table_input
                variables_table_adding_axes_unbalanced = np.asarray([variables_table_adding_axes_unbalanced], dtype=int)
                n_variables_table_adding_axes_unbalanced = 1

            if variables_table_adding_axes is None:
                variables_table_adding_axes = variables_table_adding_axes_unbalanced
            else:
                variables_table_adding_axes = np.append(
                    variables_table_adding_axes, variables_table_adding_axes_unbalanced, axis=0)

            # format axes_inserting_array_object
            n_axes_inserting_array_object = n_variables_table_adding_axes_unbalanced
            n_axes_array_object = n_axes_inserting_array_object
            try:
                n_axes_inserting_array_object = len(axes_inserting_array_object)
                axes_inserting_array_object = np.asarray(axes_inserting_array_object, dtype=int)
                axes_inserting_array_object[axes_inserting_array_object < 0] += n_axes_array_object
                # check point 2
                for a in axes_inserting_array_object:
                    if np.sum(a == axes_inserting_array_object) > 1:
                        raise ValueError('axes_inserting_array_object cannot contain repeated values')
                # check point 6
                if n_variables_table_adding_axes_unbalanced != n_axes_inserting_array_object:
                    raise ValueError('Shapes of axes_inserting_array_object and '
                                     'variables_table_adding_axes_unbalanced must be equal')
            except TypeError:
                if axes_inserting_array_object is None:
                    axes_inserting_array_object = np.arange(n_axes_inserting_array_object)
                else:
                    if axes_inserting_array_object < 0:
                        axes_inserting_array_object += n_axes_array_object
                    axes_inserting_array_object = np.asarray([axes_inserting_array_object], dtype=int)
                    n_axes_inserting_array_object = 1
                    # check point 6
                    if n_variables_table_adding_axes_unbalanced != n_axes_inserting_array_object:
                        raise ValueError('Shapes of axes_inserting_array_object and '
                                         'variables_table_adding_axes_unbalanced must be equal')

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
    else:
        n_variables_table_staying = variables_table_staying.size
        if variables_table_adding_axes_balanced is not None:
            n_variables_table_adding_axes_balanced = variables_table_adding_axes_balanced.size
        if variables_table_adding_axes_unbalanced is not None:
            n_variables_table_adding_axes_unbalanced = variables_table_adding_axes_unbalanced.size

    indexes = np.empty(n_axes_array_input, dtype=object)
    indexes[:] = slice(None)
    if ((n_variables_table_staying != n_variables_table_input) or
            np.any(variables_table_staying != variables_table_input)):
        indexes[axis_variables_table_input] = variables_table_staying
    array_variables_staying = array_input[tuple(indexes)]

    if variables_table_adding_axes_balanced is not None:
        if ((n_variables_table_adding_axes_balanced != n_variables_table_input) or
                np.any(variables_table_adding_axes_balanced != variables_table_input)):

            indexes[axis_variables_table_input] = variables_table_adding_axes_balanced
        else:
            indexes[axis_variables_table_input] = slice(None)
        array_variables_adding_axes_balanced = array_input[tuple(indexes)]

    if variables_table_adding_axes_unbalanced is not None:
        if ((n_variables_table_adding_axes_unbalanced != n_variables_table_input) or
                np.any(variables_table_adding_axes_unbalanced != variables_table_input)):

            indexes[axis_variables_table_input] = variables_table_adding_axes_unbalanced
        else:
            indexes[axis_variables_table_input] = slice(None)
        array_variables_adding_axes_unbalanced = array_input[tuple(indexes)]

    del array_input

    if variables_table_adding_axes_balanced is not None and variables_table_adding_axes_unbalanced is not None:
        # from ccalafiore.array import samples_in_arr1_are_in_arr2
        # check point 8
        if np.any(samples_in_arr1_are_in_arr2(
                variables_table_adding_axes_balanced, variables_table_adding_axes_unbalanced)):
            raise ValueError('Any value in variables_table_adding_axes_balanced must be different to\n'
                             'any value in variables_table_adding_axes_unbalanced')

        # variables_table_staying_not_object = np.unique(np.append(
        #     variables_table_staying, variables_table_adding_axes_unbalanced))

        array_variables_staying = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

        array_variables_adding_axes_unbalanced = v2a_from_2_arrays(
            array_variables_adding_axes_unbalanced,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=None,
            format_and_check=False)

        # axis_variables_table_input += np.sum(axes_inserting_array_not_object <= axis_variables_table_input)
        # axis_samples_input += np.sum(axes_inserting_array_not_object <= axis_samples_input)
        axis_variables_table_output = axis_variables_table_input + np.sum(
            axes_inserting_array_not_object <= axis_variables_table_input)
        axis_samples_output = axis_samples_input + np.sum(axes_inserting_array_not_object <= axis_samples_input)
        changed = True
        while changed:
            changed = False
            while axis_variables_table_output in axes_inserting_array_not_object:
                axis_variables_table_output += 1
                changed = True
            while axis_samples_output in axes_inserting_array_not_object:
                axis_samples_output += 1
                changed = True
            if axis_variables_table_output == axis_samples_output:
                changed = True
                if axis_variables_table_input > axis_samples_input:
                    axis_variables_table_output += 1
                elif axis_variables_table_input < axis_samples_input:
                    axis_samples_output += 1

        axis_variables_table_input = axis_variables_table_output
        axis_samples_input = axis_samples_output

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    elif variables_table_adding_axes_balanced is not None and variables_table_adding_axes_unbalanced is None:

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_balanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_not_object,
            balance=True,
            dtype=dtype,
            format_and_check=False)

    elif variables_table_adding_axes_balanced is None and variables_table_adding_axes_unbalanced is not None:

        if dtype is not None:
            array_variables_staying = array_variables_staying.astype(dtype)

        array_output = v2a_from_2_arrays(
            array_variables_staying,
            array_variables_adding_axes_unbalanced,
            axis_samples_input,
            axis_variables_table_input,
            axes_inserting=axes_inserting_array_object,
            balance=False,
            dtype=None,
            format_and_check=False)

    return array_output
