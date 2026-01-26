import numpy as np
import math
import itertools
from . import format as cp_format
from . import array as cp_array
from . import maths as cp_maths


class Conditions:

    def __init__(self, conditions):

        """

        :param conditions:
        :type conditions: list[list[object]]

        """

        if isinstance(conditions, list):
            self.conditions = conditions

        elif isinstance(conditions, tuple):
            self.conditions = list(conditions)

        elif isinstance(conditions, np.ndarray):
            self.conditions = conditions.tolist()
        else:
            raise TypeError('conditions')

        self.n_variables = len(self.conditions)
        self.n_conditions = np.asarray([len(self.conditions[v]) for v in range(0, self.n_variables, 1)], dtype='i')

        self.n_combinations = cp_maths.prod(self.n_conditions)

    def __iter__(self):

        self._i = -1
        self._combination_indexes_i = np.empty(self.n_variables, dtype='i')
        self._combination_indexes_i[:] = 0

        self._combination_values_i = [
            self.conditions[v][self._combination_indexes_i[v]] for v in range(0, self.n_variables, 1)]

        return self

    def __next__(self):

        self._i += 1

        if self._i == 0:
            return self._combination_indexes_i, self._combination_values_i

        elif self._i < self.n_combinations:

            v = self.n_variables - 1
            decrease_condition_v = True

            while decrease_condition_v:
                self._combination_indexes_i[v] += 1
                if self._combination_indexes_i[v] >= self.n_conditions[v]:
                    self._combination_indexes_i[v] = 0

                else:
                    decrease_condition_v = False

                self._combination_values_i[v] = self.conditions[v][self._combination_indexes_i[v]]
                v -= 1

            return self._combination_indexes_i, self._combination_values_i

        else:
            raise StopIteration


def n_conditions_to_combinations(
        n_conditions,
        axis_combinations=0,
        n_repetitions_combinations=1,
        order_variables='rl',
        variables_in_order=None,
        dtype='i'):

    # order_variables is the order of the variables by which they change their own conditions.
    # order_variables can either be 'rl' for right-to-left, 'lr' for left-to-right or 'c' for custom.
    # If order_variables=='c', then variables_in_order must be a list, a tuple or an 1-d array containing the variables
    # in the custom order by which the variables change their own conditions:
    #
    # Requirements:
    # 1) order_variables == 'rl' or order_variables == 'lr' or order_variables == 'c';
    # If order_variables=='c', then:
    #     2) len(variables_in_order) == len(n_conditions);
    #     3) variables_in_order must contain all integers from 0 to (len(n_conditions)-1);
    #     4) variables_in_order cannot contain repeated integers.

    n_variables = len(n_conditions)
    if n_variables > 0:
        variables = np.arange(n_variables)
        # n_conditions = np.asarray(n_conditions, dtype=int)
        # conditions = n_conditions_to_conditions(n_conditions)
        n_combinations = cp_maths.prod(n_conditions) * n_repetitions_combinations

        # axis_combinations = int(not(bool(axis_variables)))
        axis_variables = int(not(bool(axis_combinations)))
        shape_combinations = np.empty(2, dtype='i')
        shape_combinations[axis_combinations] = n_combinations
        shape_combinations[axis_variables] = n_variables

        combinations = np.empty(shape_combinations, dtype=dtype)

        if order_variables == 'rl':
            variables_in_order = variables[::-1]
        elif order_variables == 'lr':
            variables_in_order = variables
        elif order_variables == 'c':
            variables_in_order = np.asarray(variables_in_order, dtype='i')
            # check requirement 2
            if len(variables_in_order) != n_variables:
                raise ValueError('The following condition must be met:\nlen(variables_in_order) == len(n_conditions).')
            # check requirement 3
            for v in range(n_variables):
                if v not in variables_in_order:
                    raise ValueError('variables_in_order must contain all integers from 0 to (len(n_conditions)-1).\n'
                                     '{} is not in variables_in_order.'.format(v))
            # check requirement 4
            for v in variables_in_order:
                if np.sum(v == variables_in_order) > 1:
                    raise ValueError('variables_in_order cannot contain repeated integers.\n'
                                     '{} is a repeated integer.'.format(v))
        else:
            # check requirement 1
            raise ValueError(
                'order_variables can either be \'rl\' for right-to-left, '
                '\'lr\' for left-to-right or \'c\' for custom.\n'
                'Now, order_variables = {}'.format(order_variables))

        i_variable = variables_in_order[0]
        indexes_combinations = np.empty(2, dtype=object)
        indexes_combinations[axis_variables] = i_variable
        for i_condition in range(n_conditions[i_variable]):
            indexes_combinations[axis_combinations] = slice(i_condition, n_combinations, n_conditions[i_variable])
            combinations[tuple(indexes_combinations)] = i_condition

        indexes_combinations[axis_combinations] = slice(None)

        # combinations[tuple(indexes_combinations)] = np.expand_dims(
        #     cp_array.pad_array_from_n_samples_target(conditions[-1], n_samples_target=n_combinations),
        #     axis=axis_variables)
        cumulative_n_combinations = n_conditions[i_variable]
        for i_variable in variables_in_order[1:]:
            cumulative_combinations = np.empty(
                cumulative_n_combinations * n_conditions[i_variable], combinations.dtype)
            for i_condition in range(n_conditions[i_variable]):
                cumulative_combinations[
                    slice(i_condition * cumulative_n_combinations,
                          (i_condition + 1) * cumulative_n_combinations)] = i_condition

            indexes_combinations[axis_variables] = i_variable
            if cumulative_combinations.size < n_combinations:
                combinations[tuple(indexes_combinations)] = cp_array.pad_array_from_n_samples_target(
                    cumulative_combinations, n_samples_target=n_combinations)
            elif cumulative_combinations.size == n_combinations:
                combinations[tuple(indexes_combinations)] = cumulative_combinations
            else:
                raise Exception('something wrong')
            cumulative_n_combinations *= n_conditions[i_variable]

        # if change_dtype:
        #     combinations = combinations.astype(dtype_end)
    else:
        combinations = np.empty([0, 0], dtype='i')
    return combinations


def n_conditions_to_combinations_on_the_fly(
        n_conditions, order_variables='rl', variables_in_order=None, dtype='i'):
    try:
        n_variables = len(n_conditions)
        if not isinstance(n_conditions, np.ndarray):
            n_conditions = np.asarray(n_conditions, dtype='i')
    except TypeError:
        n_conditions = np.asarray([n_conditions], dtype='i')
        n_variables = 1

    n_combinations = cp_maths.prod(n_conditions)

    combination_i = np.empty(n_variables, dtype=dtype)
    combination_i[:] = 0

    yield combination_i

    for i in range(1, n_combinations):

        v = n_variables - 1
        increase_condition_v = True

        while increase_condition_v:
            combination_i[v] += 1
            if combination_i[v] >= n_conditions[v]:
                combination_i[v] = 0
                v -= 1
            else:
                increase_condition_v = False

        yield combination_i


def conditions_to_combinations_on_the_fly(
        conditions_raw, order_variables='rl', variables_in_order=None, dtype=None, order_outputs='vi'):
    """

    Parameters
    ----------
    conditions_raw
    order_variables
    variables_in_order
    dtype
    order_outputs : str or sequence of str, optional
        The desired outputs. Accepted values are "v", "i" or any combination of them like "vi", "iv" (default is "vi").
        "v" stands for combination_values_i and "i" for combination_indexes_i.

    Returns
    -------

    """

    if not isinstance(conditions_raw, np.ndarray):
        conditions = np.asarray(conditions_raw, dtype=object)
    else:
        conditions = conditions_raw

    n_conditions = conditions_to_n_conditions(conditions)
    n_combinations = cp_maths.prod(n_conditions)

    n_variables = len(n_conditions)

    combination_indexes_i = np.empty(n_variables, dtype='i')
    combination_indexes_i[:] = 0

    order_accepted_values = 'vi'
    if order_outputs is None:
        order_outputs = 'vi'
        n_outputs = 2
    else:
        n_outputs = len(order_outputs)
        if n_outputs < 1:
            raise ValueError('order_outputs')
        else:
            for o in range(n_outputs):
                if not (order_outputs[o] in order_accepted_values):
                    raise ValueError('order_outputs')

    outputs_i = [None] * n_outputs  # type: list

    for o in range(n_outputs):

        if order_outputs[o] == 'v':

            if dtype is None:
                combination_values_i = [None] * n_variables
                dtype_np = None
            else:
                dtype_np = np.dtype(dtype)
                if dtype_np.kind == 'U':
                    combination_values_i = np.empty(n_variables, dtype='O')
                else:
                    combination_values_i = np.empty(n_variables, dtype=dtype)

            for v in range(n_variables):
                combination_values_i[v] = conditions[v][combination_indexes_i[v]]

            if dtype_np is None:
                combination_values_i = np.asarray(combination_values_i)
                dtype_np = combination_values_i.dtype
                if dtype_np.kind == 'U':
                    combination_values_i = combination_values_i.astype('O')

            if dtype_np.kind == 'U':
                outputs_i[o] = combination_values_i.astype('U')
            else:
                outputs_i[o] = combination_values_i

        elif order_outputs[o] == 'i':

            outputs_i[o] = combination_indexes_i

    if n_outputs == 1:
        yield outputs_i[0]
    else:
        yield outputs_i

    for i in range(1, n_combinations):

        v = n_variables - 1
        increase_condition_v = True

        while increase_condition_v:
            combination_indexes_i[v] += 1
            if combination_indexes_i[v] >= n_conditions[v]:
                combination_indexes_i[v] = 0
                v -= 1
            else:
                increase_condition_v = False

        for o in range(n_outputs):

            if order_outputs[o] == 'v':

                for v in range(n_variables):
                    combination_values_i[v] = conditions[v][combination_indexes_i[v]]

                if dtype_np.kind == 'U':
                    outputs_i[o] = combination_values_i.astype('U')
                else:
                    outputs_i[o] = combination_values_i

            elif order_outputs[o] == 'i':

                outputs_i[o] = combination_indexes_i

        if n_outputs == 1:
            yield outputs_i[0]
        else:
            yield outputs_i


# def conditions_to_combinations_on_the_fly(
#         conditions, order_variables='rl', variables_in_order=None, dtype=None):
#
#     for _, combination_i in conditions_to_combinations_on_the_fly_with_indexes(
#             conditions, order_variables=order_variables,
#             variables_in_order=variables_in_order, dtype=dtype):
#
#         yield combination_i


def conditions_to_combinations(
        conditions,
        axis_combinations=0,
        n_repetitions_combinations=1,
        order_variables='rl',
        variables_in_order=None,
        dtype=None):

    # order_variables is the order of the variables by which they change their own conditions.
    # order_variables can either be 'rl' for right-to-left, 'lr' for left-to-right or 'c' for custom.
    # If order_variables=='c', then variables_in_order must be a list, a tuple or an 1-d array containing the variables
    # in the custom order by which the variables change their own conditions:
    #
    # Requirements:
    # 1) order_variables == 'rl' or order_variables == 'lr' or order_variables == 'c';
    # If order_variables=='c', then:
    #     2) len(variables_in_order) == len(conditions);
    #     3) variables_in_order contains all integers from 0 to (len(conditions)-1);
    #     4) variables_in_order cannot contain repeated integers.

    n_variables = len(conditions)
    if n_variables > 0:
        variables = np.arange(n_variables)
        # conditions = np.asarray(conditions, dtype=object)
        n_conditions = conditions_to_n_conditions(conditions)
        n_combinations = cp_maths.prod(n_conditions) * n_repetitions_combinations

        if dtype is None:
            dtype = type(conditions[0][0])

        change_dtype = False
        if dtype == str:
            dtype_end = str
            dtype = object
            change_dtype = True

        axis_variables = int(not(bool(axis_combinations)))
        shape_combinations = np.empty(2, dtype=int)
        shape_combinations[axis_combinations] = n_combinations
        shape_combinations[axis_variables] = n_variables

        combinations = np.empty(shape_combinations, dtype=dtype)

        if order_variables == 'rl':
            variables_in_order = variables[::-1]
        elif order_variables == 'lr':
            variables_in_order = variables
        elif order_variables == 'c':
            variables_in_order = np.asarray(variables_in_order, dtype=int)
            # check requirement 2
            if len(variables_in_order) != n_variables:
                raise ValueError('The following condition must be met:\nlen(variables_in_order) == len(conditions).')
            # check requirement 3
            for v in range(n_variables):
                if v not in variables_in_order:
                    raise ValueError('variables_in_order must contain all integers from 0 to (len(conditions)-1).\n'
                                     '{} is not in variables_in_order.'.format(v))
            # check requirement 4
            for v in variables_in_order:
                if np.sum(v == variables_in_order) > 1:
                    raise ValueError('variables_in_order cannot contain repeated integers.\n'
                                     '{} is a repeated integer.'.format(v))
        else:
            # check requirement 1
            raise ValueError(
                'order_variables can either be \'rl\' for right-to-left, '
                '\'lr\' for left-to-right or \'c\' for custom.\n'
                'Now, order_variables = {}'.format(order_variables))

        i_variable = variables_in_order[0]
        indexes_combinations = np.empty(2, dtype=object)
        indexes_combinations[axis_variables] = i_variable
        for i_condition in range(n_conditions[i_variable]):
            indexes_combinations[axis_combinations] = slice(i_condition, n_combinations, n_conditions[i_variable])
            combinations[tuple(indexes_combinations)] = conditions[i_variable][i_condition]

        indexes_combinations[axis_combinations] = slice(None)

        # combinations[tuple(indexes_combinations)] = np.expand_dims(
        #     cp_array.pad_array_from_n_samples_target(conditions[-1], n_samples_target=n_combinations),
        #     axis=axis_variables)
        cumulative_n_combinations = n_conditions[i_variable]
        for i_variable in variables_in_order[1:]:
            cumulative_combinations = np.empty(
                cumulative_n_combinations * n_conditions[i_variable], combinations.dtype)
            for i_condition in range(n_conditions[i_variable]):
                cumulative_combinations[
                    slice(i_condition * cumulative_n_combinations,
                          (i_condition + 1) * cumulative_n_combinations)] = conditions[i_variable][i_condition]

            indexes_combinations[axis_variables] = i_variable
            if cumulative_combinations.size < n_combinations:
                combinations[tuple(indexes_combinations)] = cp_array.pad_array_from_n_samples_target(
                    cumulative_combinations, n_samples_target=n_combinations)
            elif cumulative_combinations.size == n_combinations:
                combinations[tuple(indexes_combinations)] = cumulative_combinations
            else:
                raise Exception('something wrong')
            cumulative_n_combinations *= n_conditions[i_variable]

        if change_dtype:
            combinations = combinations.astype(dtype_end)
    else:
        combinations = np.empty([0, 0], dtype='i')
    return combinations


def make_combinations_of_conditions_as_distributions(conditions_as_distributions, n_repetitions_combinations=1):

    n_variables = len(conditions_as_distributions)
    n_conditions = np.empty(n_variables, dtype=object)

    for i_variable in range(n_variables):

        n_conditions[i_variable] = len(conditions_as_distributions[i_variable])

    combinations_of_conditions = n_conditions_to_combinations(n_conditions)

    n_combinations = len(combinations_of_conditions)

    combinations_of_conditions_as_distributions = np.empty([n_combinations, n_variables], dtype=int)

    for i_variable in range(n_variables):

        for i_condition in range(n_conditions[i_variable]):

            indexes_of_i_condition = np.argwhere(combinations_of_conditions[:, i_variable] == i_condition)

            n_i_condition = len(indexes_of_i_condition)

            combinations_of_conditions_as_distributions[indexes_of_i_condition, i_variable] = \
                np.random.choice(conditions_as_distributions[i_variable][i_condition],
                                 n_i_condition)[:, None]

    return combinations_of_conditions_as_distributions


def trials_to_combinations(
        trials, axis_combinations=0, variables=None, n_repetitions_combinations=1, dtype=None,
        exclude_values=False, values=-1):

    conditions = trials_to_conditions(
        trials, axis_combinations=axis_combinations, variables=variables,
        exclude_values=exclude_values, values=values)

    combinations = conditions_to_combinations(
        conditions,
        axis_combinations=axis_combinations,
        n_repetitions_combinations=n_repetitions_combinations,
        dtype=dtype)
    return combinations


def trials_to_conditions(trials, axis_combinations=0, variables=None, exclude_values=False, values=-1):

    axis_variables = int(not(bool(axis_combinations)))
    if variables is None:
        n_variables = trials.shape[axis_variables]
        variables = np.arange(n_variables)
    else:
        try:
            n_variables = len(variables)
        except TypeError:
            variables = np.asarray([variables], dtype=int)
            n_variables = len(variables)

    conditions = np.empty(n_variables, dtype=object)

    indexes_trials = np.empty(2, dtype=object)
    indexes_trials[axis_combinations] = slice(0, trials.shape[axis_combinations], 1)

    for i_variable in range(n_variables):

        indexes_trials[axis_variables] = variables[i_variable]
        trials_variables_i = trials[tuple(indexes_trials)]

        conditions[i_variable] = np.unique(trials_variables_i)

    if exclude_values:
        conditions = exclude(conditions, values)

    return conditions


def trials_to_n_conditions(trials, axis_combinations=0, variables=None, exclude_values=False, values=-1):
    conditions = trials_to_conditions(
        trials, axis_combinations=axis_combinations, variables=variables,
        exclude_values=exclude_values, values=values)

    n_conditions = conditions_to_n_conditions(conditions)
    # n_variables = len(conditions)
    # n_conditions = np.empty(n_variables, dtype=int)
    # for i_variable in range(n_variables):
    #     n_conditions[i_variable] = len(conditions[i_variable])
    return n_conditions


def n_conditions_to_conditions(n_conditions, exclude_values=False, values=-1):
    n_variables = len(n_conditions)
    conditions = np.empty(n_variables, dtype=object)
    for i_variable in range(n_variables):
        conditions[i_variable] = np.arange(n_conditions[i_variable])

    if exclude_values:
        conditions = exclude(conditions, values)

    return conditions


def conditions_to_n_conditions(conditions):
    n_variables = len(conditions)
    n_conditions = np.empty(n_variables, dtype=int)
    for i_variable in range(n_variables):
        n_conditions[i_variable] = len(conditions[i_variable])
    return n_conditions


def exclude(conditions, values=-1, variables=None):
    n_variables = len(conditions)

    if variables is None:
        variables = list(range(n_variables))

    for i_variable in range(n_variables):
        if i_variable in variables:
            conditions[i_variable] = conditions[i_variable][
                cp_array.samples_in_arr1_are_not_in_arr2(conditions[i_variable], values)]
    return conditions


def conditions_to_phases_of_blocks_of_shuffled_combinations(
        conditions, n_repetitions_combinations=1, n_repetitions_variables_repeated_per_block=1,
        variables_split=None, n_repetitions_variables_repeated_per_shuffle=None):

    n_phases = len(conditions)

    dict_arguments = dict(
        n_repetitions_combinations=n_repetitions_combinations,
        n_repetitions_variables_repeated_per_block=n_repetitions_variables_repeated_per_block,
        variables_split=variables_split,
        n_repetitions_variables_repeated_per_shuffle=n_repetitions_variables_repeated_per_shuffle)

    dict_arguments = cp_format.format_shape_arguments(dict_arguments, n_phases)
    n_repetitions_combinations = dict_arguments['n_repetitions_combinations']
    n_repetitions_variables_repeated_per_block = dict_arguments['n_repetitions_variables_repeated_per_block']
    variables_split = dict_arguments['variables_split']
    n_repetitions_variables_repeated_per_shuffle = dict_arguments['n_repetitions_variables_repeated_per_shuffle']

    # dict_arguments = cp_format.format_shape_arguments(dict_arguments, n_phases)
    # locals().update(dict_arguments)

    phases_of_blocks_of_shuffled_combinations = np.empty(n_phases, dtype=object)

    for i_phase in range(n_phases):

        phases_of_blocks_of_shuffled_combinations[i_phase] = conditions_to_blocks_of_shuffled_combinations(
            conditions[i_phase],
            n_repetitions_combinations=n_repetitions_combinations[i_phase],
            n_repetitions_variables_repeated_per_block=n_repetitions_variables_repeated_per_block[i_phase],
            n_repetitions_variables_repeated_per_shuffle=n_repetitions_variables_repeated_per_shuffle[i_phase],
            variables_split=variables_split[i_phase])

    return phases_of_blocks_of_shuffled_combinations


def n_conditions_to_phases_of_blocks_of_shuffled_combinations(
        n_conditions, n_repetitions_combinations=1, n_repetitions_variables_repeated_per_block=1,
        variables_split=None, n_repetitions_variables_repeated_per_shuffle=None):

    n_phases = len(n_conditions)

    dict_arguments = dict(
        n_repetitions_combinations=n_repetitions_combinations,
        n_repetitions_variables_repeated_per_block=n_repetitions_variables_repeated_per_block,
        variables_split=variables_split,
        n_repetitions_variables_repeated_per_shuffle=n_repetitions_variables_repeated_per_shuffle)

    dict_arguments = cp_format.format_shape_arguments(dict_arguments, n_phases)

    n_repetitions_combinations = dict_arguments['n_repetitions_combinations']
    n_repetitions_variables_repeated_per_block = dict_arguments['n_repetitions_variables_repeated_per_block']
    variables_split = dict_arguments['variables_split']
    n_repetitions_variables_repeated_per_shuffle = dict_arguments['n_repetitions_variables_repeated_per_shuffle']

    phases_of_blocks_of_shuffled_combinations = np.empty(n_phases, dtype=object)

    for i_phase in range(n_phases):
        phases_of_blocks_of_shuffled_combinations[i_phase] = n_conditions_to_blocks_of_shuffled_combinations(
            n_conditions[i_phase],
            n_repetitions_combinations=n_repetitions_combinations[i_phase],
            n_repetitions_variables_repeated_per_block=n_repetitions_variables_repeated_per_block[i_phase],
            n_repetitions_variables_repeated_per_shuffle=n_repetitions_variables_repeated_per_shuffle[i_phase],
            variables_split=variables_split[i_phase])

    return phases_of_blocks_of_shuffled_combinations


def n_conditions_to_blocks_of_shuffled_combinations(
        n_conditions, n_repetitions_combinations=1,
        n_repetitions_variables_repeated_per_block=1,
        n_repetitions_variables_repeated_per_shuffle=None,
        variables_split=None):

    conditions = n_conditions_to_conditions(n_conditions)

    trials_blocks = conditions_to_blocks_of_shuffled_combinations(
        conditions, n_repetitions_combinations=n_repetitions_combinations,
        n_repetitions_variables_repeated_per_block=n_repetitions_variables_repeated_per_block,
        n_repetitions_variables_repeated_per_shuffle=n_repetitions_variables_repeated_per_shuffle,
        variables_split=variables_split)

    return trials_blocks


def conditions_to_blocks_of_shuffled_combinations(
        conditions, n_repetitions_combinations=1,
        n_repetitions_variables_repeated_per_block=1,
        n_repetitions_variables_repeated_per_shuffle=None,
        variables_split=None):

    if type(conditions) != np.ndarray:
        conditions = np.array(conditions, dtype=object)

    n_conditions = conditions_to_n_conditions(conditions)

    axis_combinations = 0

    trials = conditions_to_combinations(
        conditions, axis_combinations=axis_combinations,
        n_repetitions_combinations=n_repetitions_combinations, order_variables='rl',
        variables_in_order=None, dtype=None)

    n_trials = trials.shape[axis_combinations]

    n_variables = conditions.shape[0]
    variables = np.arange(n_variables)

    try:
        n_variables_split = len(variables_split)
        if not isinstance(variables_split, np.ndarray):
            variables_split = np.asarray(variables_split, dtype=int)
    except TypeError:
        if variables_split is None:
            n_variables_split = 0
        else:
            variables_split = np.asarray([variables_split], dtype=int)
            n_variables_split = 1

    if n_variables_split == 0:

        conditions_variables_repeated = conditions
        n_conditions_variables_repeated = n_conditions
        if n_repetitions_variables_repeated_per_shuffle is None:
            n_repetitions_variables_repeated_per_shuffle = n_repetitions_variables_repeated_per_block
    else:
        variables_split[variables_split < 0] += n_variables
        n_repetitions_variables_repeated_per_shuffle = n_repetitions_variables_repeated_per_block

        if np.any(np.logical_or(variables_split < 0, variables_split >= n_variables)):
            raise Exception('There is at least 1 variables_split that is not in the variables.\n'
                            'each sample in the variables_split has to be equal to either \n'
                            'one or more samples of variables or None.\n'
                            'Now, variables_split = {} and variables = {}.\n'
                            'By default, variables_split=None.'.format(
                                variables_split, variables))

        variables_repeated = np.where([i not in variables_split for i in variables])[0]
        conditions_variables_repeated = conditions[variables_repeated]
        n_conditions_variables_repeated = n_conditions[variables_repeated]

        conditions_variables_split = conditions[variables_split]
        n_conditions_variables_split = n_conditions[variables_split]
        n_combinations_variables_split = cp_maths.prod(n_conditions_variables_split)

    n_combinations_variables_repeated = cp_maths.prod(n_conditions_variables_repeated)

    n_trials_variables_repeated = n_combinations_variables_repeated * n_repetitions_combinations

    n_trials_per_block = n_combinations_variables_repeated * n_repetitions_variables_repeated_per_block

    if n_trials % n_trials_per_block != 0:
        possibles_n_repetitions_variables_repeated_per_block = cp_maths.factors_of_x(
            n_trials // n_combinations_variables_repeated)

        raise Exception(
            'n_trials_per_blocks are not equal. Change the n_repetitions_variables_repeated_per_block\n'
            'or the n_repetitions_combinations to make the n_trials_per_blocks equal.\n'
            'Given n_repetitions_combinations = {} and n_combinations_variables_repeated = {},\n'
            'n_trials = {} and n_repetitions_variables_repeated_per_block can either\n'
            'be one of the following numbers {}. Now, n_repetitions_variables_repeated_per_block = {}.'.format(
                n_repetitions_combinations, n_combinations_variables_repeated, n_trials,
                possibles_n_repetitions_variables_repeated_per_block,
                n_repetitions_variables_repeated_per_block))

    n_blocks = n_trials // n_trials_per_block

    n_trials_per_shuffle = n_combinations_variables_repeated * n_repetitions_variables_repeated_per_shuffle

    if n_trials % n_trials_per_shuffle != 0:

        possibles_n_repetitions_variables_repeated_per_shuffle = cp_maths.factors_of_x(
            n_trials // n_combinations_variables_repeated)

        raise Exception(
            'n_trials_per_shuffles are not equal. Change the n_repetitions_variables_repeated_per_shuffle\n'
            'or the n_repetitions_combinations to make the n_trials_per_shuffles equal.\n'
            'Given n_repetitions_combinations = {} and n_combinations_variables_repeated = {},\n'
            'n_trials = {} and n_repetitions_variables_repeated_per_shuffle can either\n'
            'be one of the following numbers {}. Now, n_repetitions_variables_repeated_per_shuffle = {}.'.format(
                n_repetitions_combinations, n_combinations_variables_repeated, n_trials,
                possibles_n_repetitions_variables_repeated_per_shuffle,
                n_repetitions_variables_repeated_per_shuffle))

    n_shuffles = int(n_trials / n_trials_per_shuffle)

    if n_variables_split == 0:

        trials_blocks = np.split(trials, n_blocks, axis=axis_combinations)

    else:

        trials_1_repetition_variables_repeated = conditions_to_combinations(
            conditions_variables_repeated, axis_combinations=0, n_repetitions_combinations=1,
            order_variables='rl', variables_in_order=None, dtype=None)

        trials_1_repetition_variables_split = conditions_to_combinations(
            conditions_variables_split, axis_combinations=0, n_repetitions_combinations=1,
            order_variables='rl', variables_in_order=None, dtype=None)

        range_combinations_variables_repeated = np.arange(n_combinations_variables_repeated)

        min_trials_of_1_single_variable_split_per_block = np.empty(n_variables_split, dtype=int)
        available_conditions_variables_split_per_block = np.empty(n_variables_split, dtype=object)
        extra_available_conditions_variables_split = np.empty(n_variables_split, dtype=object)
        n_available_conditions_variables_split_per_block = np.empty(n_variables_split, dtype=int)

        for i_variable_split in range(n_variables_split):

            n_conditions_variables_split_not_in_variables_split = n_conditions_variables_split[
                np.arange(n_variables_split) != i_variable_split]

            min_trials_of_1_single_variable_split_per_block[i_variable_split] = \
                (n_combinations_variables_repeated * cp_maths.prod(
                    n_conditions_variables_split_not_in_variables_split) *
                 n_repetitions_combinations) // n_blocks

            available_conditions_variables_split_per_block[i_variable_split] = np.arange(
                n_conditions_variables_split[i_variable_split] *
                min_trials_of_1_single_variable_split_per_block[i_variable_split]) // \
                min_trials_of_1_single_variable_split_per_block[i_variable_split]

            extra_available_conditions_variables_split[i_variable_split] = np.arange(
                n_conditions_variables_split[i_variable_split])

            n_available_conditions_variables_split_per_block[i_variable_split] = len(
                available_conditions_variables_split_per_block[i_variable_split])

        n_extra_conditions_variables_split = n_trials_per_block % n_conditions_variables_split

        i_trial_block = np.empty(shape=[1, n_variables], dtype=int)  # TO MOVE BEFORE THE FOR LOOPS
        # k = 0
        u = 0
        block = True
        while block:
            u += 1
            print('Block {}'.format(u))

            block = False

            tmp_trials = trials
            shape_frequencies = n_conditions_variables_split
            frequencies_combinations_repeated_and_split = np.zeros(
                (n_combinations_variables_repeated, *n_conditions_variables_split), dtype=int)

            extra_available_conditions_variables_split_per_i_block = np.copy(extra_available_conditions_variables_split)

            available_conditions_variables_split_per_i_block = np.empty(n_blocks, object)
            trials_blocks = np.empty(n_blocks, dtype=object)

            for i_block in range(n_blocks):

                available_conditions_variables_split_per_i_block[i_block] = np.copy(
                    available_conditions_variables_split_per_block)

                n_available_conditions_variables_split_per_i_block = np.empty(n_variables_split, dtype=int)

                for i_variable_split in range(n_variables_split):

                    if n_extra_conditions_variables_split[i_variable_split] > 0:

                        for i_extra_condition_variables_split in range(
                                n_extra_conditions_variables_split[i_variable_split]):

                            n_extra_available_conditions_variables_split_per_i_block = len(
                                extra_available_conditions_variables_split_per_i_block[i_variable_split])

                            if n_extra_available_conditions_variables_split_per_i_block < 1:
                                extra_available_conditions_variables_split_per_i_block[i_variable_split] = \
                                    np.copy(extra_available_conditions_variables_split[i_variable_split])

                            extra_available_conditions_variables_split_per_i_block[i_variable_split], \
                                available_conditions_variables_split_per_i_block[i_block][i_variable_split] = \
                                cp_array.transfer_n_random_samples_from_arr1_to_arr2(
                                    extra_available_conditions_variables_split_per_i_block[i_variable_split],
                                    available_conditions_variables_split_per_i_block[i_block][i_variable_split],
                                    n_samples=1, axis=0, replace=False)

                    n_available_conditions_variables_split_per_i_block[i_variable_split] = len(
                        available_conditions_variables_split_per_i_block[i_block][i_variable_split])

                if np.any(n_available_conditions_variables_split_per_i_block[0] !=
                          n_available_conditions_variables_split_per_i_block):
                    raise Warning(
                        'lengths of n_available_conditions_variables_split_per_i_block are not all equal.\n'
                        'n_available_conditions_variables_split_per_i_block = {}'.format(
                            n_available_conditions_variables_split_per_block))

                trials_blocks[i_block] = np.empty((0, n_variables), int)

                for i_combinations_variables_repeated in range(n_combinations_variables_repeated):

                    i_repetition_variables_repeated_per_block = 0
                    while i_repetition_variables_repeated_per_block < n_repetitions_variables_repeated_per_block:

                        available_combinations_variables_split_per_i_trial = np.argwhere(
                            frequencies_combinations_repeated_and_split[
                                i_combinations_variables_repeated] <
                            n_repetitions_combinations)

                        tmp_logical_index_available_combinations_variables_split_per_i_block_per_i_trial = np.empty(
                            (len(available_combinations_variables_split_per_i_trial), n_variables_split), dtype=bool)

                        for i_variable_split in range(n_variables_split):
                            tmp_logical_index_available_combinations_variables_split_per_i_block_per_i_trial[
                                slice(None), i_variable_split] = cp_array.samples_in_arr1_are_in_arr2(
                                    available_combinations_variables_split_per_i_trial[:, i_variable_split],
                                    available_conditions_variables_split_per_i_block[i_block][i_variable_split])

                        logical_indexes_available_combinations_variables_split_per_i_block_per_i_trial = np.all(
                            tmp_logical_index_available_combinations_variables_split_per_i_block_per_i_trial, axis=1)

                        available_combinations_variables_split_per_i_block_per_i_trial = \
                            available_combinations_variables_split_per_i_trial[
                                logical_indexes_available_combinations_variables_split_per_i_block_per_i_trial]

                        n_available_combinations_variables_split_per_i_block_per_i_trial = len(
                            available_combinations_variables_split_per_i_block_per_i_trial)

                        if n_available_combinations_variables_split_per_i_block_per_i_trial > 0:

                            index_of_chosen_combination_variables_split = np.random.randint(
                                n_available_combinations_variables_split_per_i_block_per_i_trial)

                            chosen_combination_variables_split = np.expand_dims(
                                available_combinations_variables_split_per_i_block_per_i_trial[
                                    index_of_chosen_combination_variables_split], axis=0)

                            i_trial_block[0, variables_repeated] = trials_1_repetition_variables_repeated[
                                i_combinations_variables_repeated]

                            for i_variable_split in range(n_variables_split):
                                i_trial_block[0, variables_split[i_variable_split]] = conditions_variables_split[i_variable_split][
                                    chosen_combination_variables_split[0, i_variable_split]]

                            trials_blocks[i_block] = np.append(trials_blocks[i_block], i_trial_block, axis=0)

                            indexes_frequencies_combinations = np.empty(n_variables_split + 1, object)
                            for i_variable_split in range(n_variables_split + 1):
                                if i_variable_split == 0:
                                    indexes_frequencies_combinations[i_variable_split] = np.asarray(
                                        [i_combinations_variables_repeated])
                                else:
                                    indexes_frequencies_combinations[i_variable_split] = np.asarray(
                                        [chosen_combination_variables_split[0, i_variable_split - 1]])

                            frequencies_combinations_repeated_and_split[np.ix_(*indexes_frequencies_combinations)] += 1

                            for i_variable_split in range(n_variables_split):

                                first_index_chosen_conditions_variables_split = np.argwhere(
                                    available_conditions_variables_split_per_i_block[i_block][i_variable_split] == \
                                    chosen_combination_variables_split[0, i_variable_split])[0]

                                available_conditions_variables_split_per_i_block[i_block][i_variable_split] = np.delete(
                                    available_conditions_variables_split_per_i_block[i_block][i_variable_split],
                                    first_index_chosen_conditions_variables_split,  axis=0)

                            logical_index_i_trial = np.all(tmp_trials == i_trial_block, axis=1)
                            index_i_trial = np.argwhere(logical_index_i_trial)[0]

                            tmp_trials = np.delete(tmp_trials, index_i_trial, 0)
                            i_repetition_variables_repeated_per_block += 1

                        else:
                            # which trial in this block I can put the last actors available for this block that are not
                            # available for this trial

                            tmp_conditions_variables_split = np.empty(n_variables_split, dtype=object)
                            tmp_extra_conditions_variables_split = np.empty(n_variables_split, dtype=object)
                            j = 0
                            i = 0
                            for i_variable_split in range(n_variables_split):

                                if len(available_conditions_variables_split_per_i_block[
                                        i_block][i_variable_split]) > 0:

                                    tmp_conditions_variables_split[j] = np.unique(
                                        available_conditions_variables_split_per_i_block[i_block][i_variable_split])
                                    j += 1
                                else:
                                    tmp_conditions_variables_split = np.delete(
                                        tmp_conditions_variables_split, i_variable_split, axis=0)

                                if len(extra_available_conditions_variables_split_per_i_block[i_variable_split]) > 0:

                                    tmp_extra_conditions_variables_split[i] = \
                                        extra_available_conditions_variables_split_per_i_block[i_variable_split]
                                    i += 1
                                else:
                                    tmp_extra_conditions_variables_split = np.delete(
                                        tmp_extra_conditions_variables_split, i, axis=0)

                            if len(tmp_conditions_variables_split) > 1:
                                tmp_combinations_variables_split = conditions_to_combinations(
                                    tmp_conditions_variables_split, axis_combinations=0,
                                    n_repetitions_combinations=1, order_variables='rl',
                                    variables_in_order=None, dtype=None)
                            else:
                                tmp_combinations_variables_split = np.array(list(
                                    tmp_conditions_variables_split), dtype=int)

                            if len(tmp_extra_conditions_variables_split) > 1:
                                tmp_combinations_extra_conditions_variables_split = (
                                    conditions_to_combinations(
                                        tmp_extra_conditions_variables_split, axis_combinations=0,
                                        n_repetitions_combinations=1, order_variables='rl',
                                        variables_in_order=None, dtype=None))
                            else:
                                tmp_combinations_extra_conditions_variables_split = np.array(
                                    list(tmp_extra_conditions_variables_split), dtype=int)

                            n_tries = 2
                            try_available_combinations_variables_split_for_i_block = np.empty(
                                n_tries, dtype=object)

                            try_available_combinations_variables_split_for_i_block[0] = \
                                np.random.permutation(tmp_combinations_variables_split)

                            try_available_combinations_variables_split_for_i_block[1] = \
                                np.random.permutation(tmp_combinations_extra_conditions_variables_split)

                            try_available_combinations_variables_split_for_i_trial = np.random.permutation(
                                available_combinations_variables_split_per_i_trial)

                            n_available_combinations_variables_split_for_i_trial = len(
                                available_combinations_variables_split_per_i_trial)

                            search_break = False

                            for i_try in range(n_tries):

                                n_try_available_combinations_variables_split_for_i_block = \
                                    try_available_combinations_variables_split_for_i_block[i_try].shape[0]

                                for i_combinations_variables_split_for_i_block in range(
                                        n_try_available_combinations_variables_split_for_i_block):

                                    for i_combinations_variables_split_for_i_trial in range(
                                            n_available_combinations_variables_split_for_i_trial):

                                        indexes_frequencies_combinations = np.empty(n_variables_split + 1, object)
                                        for i_variable_split in range(n_variables_split + 1):
                                            if i_variable_split == 0:
                                                indexes_frequencies_combinations[i_variable_split] = \
                                                    range_combinations_variables_repeated
                                            else:
                                                indexes_frequencies_combinations[i_variable_split] = np.asarray(
                                                    [try_available_combinations_variables_split_for_i_block[
                                                    i_try][i_combinations_variables_split_for_i_block,
                                                           i_variable_split - 1]])

                                        logical_indexes_available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle = np.squeeze(
                                            frequencies_combinations_repeated_and_split[np.ix_(*indexes_frequencies_combinations)] <
                                            n_repetitions_combinations)

                                        available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle = trials_1_repetition_variables_repeated[
                                            logical_indexes_available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle]

                                        n_available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle = available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle.shape[0]

                                        n_2 = trials_blocks[i_block].shape[0]
                                        log2 = np.empty(shape=(n_2, n_available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle), dtype=bool)
                                        for l in range(n_available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle):
                                            log2[:, l] = np.all(
                                                trials_blocks[i_block][:, variables_repeated] ==
                                                available_combinations_variables_repeated_for_i_try_compination_split_from_i_shaffle[l, :], axis=1)

                                        log3 = np.any(log2, axis=1)

                                        inx = np.empty(n_variables_split, dtype=int)
                                        for i_variable_split in range(n_variables_split):
                                            inx[i_variable_split] = (
                                                conditions_variables_split[i_variable_split][
                                                    try_available_combinations_variables_split_for_i_trial[
                                                        i_combinations_variables_split_for_i_trial, i_variable_split]])

                                        log4 = np.all(trials_blocks[i_block][:, variables_split] == inx, axis=1)

                                        a = try_available_combinations_variables_split_for_i_trial[
                                            i_combinations_variables_split_for_i_trial]

                                        log5 = np.all(np.array([log3, log4]), axis=0)

                                        if np.sum(log5) > 0:

                                            inx2 = np.empty(n_variables_split, dtype=int)
                                            for i_variable_split in range(n_variables_split):
                                                inx2[i_variable_split] = conditions_variables_split[
                                                    i_variable_split][
                                                    try_available_combinations_variables_split_for_i_block[
                                                        i_try][i_combinations_variables_split_for_i_block, i_variable_split]]

                                            chosen_combination_variables_split_from_i_block = inx2
                                            indexes_chosen_combination_variables_split_from_i_block = \
                                                try_available_combinations_variables_split_for_i_block[
                                                    i_try][i_combinations_variables_split_for_i_block]

                                            chosen_combination_variables_split_for_i_trial = inx
                                            indexes_chosen_combination_variables_split_for_i_trial = \
                                                try_available_combinations_variables_split_for_i_trial[
                                                    i_combinations_variables_split_for_i_trial]

                                            if i_try == 0:
                                                search_break = True
                                                break

                                            elif i_try == 1:

                                                try:

                                                    conditions_variables_split_leaving_i_block = np.empty(n_variables_split, dtype=int)

                                                    for i_variable_split in range(n_variables_split):

                                                        conditions_variables_split_leaving_i_block[i_variable_split] = \
                                                            np.random.choice(np.setdiff1d(
                                                                available_conditions_variables_split_per_i_block[
                                                                    i_block][i_variable_split],
                                                                extra_available_conditions_variables_split_per_i_block[
                                                                    i_variable_split]), 1)

                                                        first_index_leaving = np.argwhere(
                                                            available_conditions_variables_split_per_i_block[
                                                                i_block][i_variable_split]
                                                            == conditions_variables_split_leaving_i_block[i_variable_split])[0, 0]

                                                        available_conditions_variables_split_per_i_block[
                                                            i_block][i_variable_split][first_index_leaving] = \
                                                            indexes_chosen_combination_variables_split_from_i_block[
                                                                i_variable_split]

                                                        extra_available_conditions_variables_split_per_i_block[
                                                            i_variable_split][
                                                            extra_available_conditions_variables_split_per_i_block[
                                                            i_variable_split] ==
                                                            indexes_chosen_combination_variables_split_from_i_block[
                                                                i_variable_split]] = conditions_variables_split_leaving_i_block

                                                except ValueError:
                                                    pass

                                        if i_try == n_tries - 1:
                                            if i_combinations_variables_split_for_i_block == \
                                                    n_try_available_combinations_variables_split_for_i_block - 1 \
                                                    and i_combinations_variables_split_for_i_trial == \
                                                    n_available_combinations_variables_split_for_i_trial - 1:
                                                block = True
                                                search_break = True
                                                break



                                    if search_break:
                                        break

                                if search_break:
                                    search_break = False
                                    break

                            if block:
                                break

                            c = np.reshape(np.argwhere(log5), (-1))


                            chosen_trial = np.random.choice(c, 1)

                            trials_blocks[i_block][
                                chosen_trial, variables_split] = chosen_combination_variables_split_from_i_block


                            ddds = np.argwhere(np.all(
                                tmp_trials == trials_blocks[i_block][chosen_trial, :],
                                axis=1))[0]
                            #

                            tmp_trials[ddds, variables_split] = chosen_combination_variables_split_for_i_trial

                            index_l = np.all(
                                trials_blocks[i_block][chosen_trial, variables_repeated] ==
                                trials_1_repetition_variables_repeated, axis=1)

                            if np.sum(index_l) != 1:
                                raise Exception(
                                    'sum of index_l should either be 1. Now it was {}.'.format(np.sum(index_l)))
                            index_l = np.argwhere(index_l)[0]

                            indexes_frequencies_combinations = np.empty(n_variables_split + 1, object)
                            for i_variable_split in range(n_variables_split + 1):
                                if i_variable_split == 0:
                                    indexes_frequencies_combinations[i_variable_split] = index_l
                                else:
                                    indexes_frequencies_combinations[i_variable_split] = np.asarray(
                                        [indexes_chosen_combination_variables_split_for_i_trial[
                                            i_variable_split - 1]])

                            frequencies_combinations_repeated_and_split[np.ix_(*indexes_frequencies_combinations)] -= 1

                            indexes_frequencies_combinations = np.empty(n_variables_split + 1, object)
                            for i_variable_split in range(n_variables_split + 1):
                                if i_variable_split == 0:
                                    indexes_frequencies_combinations[i_variable_split] = index_l
                                else:
                                    indexes_frequencies_combinations[i_variable_split] = np.asarray(
                                        [indexes_chosen_combination_variables_split_from_i_block[
                                            i_variable_split - 1]])

                            frequencies_combinations_repeated_and_split[np.ix_(*indexes_frequencies_combinations)] += 1

                            for i_variable_split in range(n_variables_split):

                                first_index_chosen_conditions_variables_split = np.argwhere(
                                    available_conditions_variables_split_per_i_block[i_block][i_variable_split] ==
                                    indexes_chosen_combination_variables_split_from_i_block[i_variable_split])[0]

                                available_conditions_variables_split_per_i_block[i_block][i_variable_split][
                                    first_index_chosen_conditions_variables_split] = \
                                    indexes_chosen_combination_variables_split_for_i_trial[i_variable_split]

                    if block:
                        break

                if block:
                    break

                if len(trials_blocks[i_block]) != n_trials_per_block:
                    raise Exception('the n trials in trials_blocks[i_block] should be equal to n_trials_per_block. '
                                    'n trials in trials_blocks[i_block] was {} whereas n_trials_per_block was {}'
                                    '.'.format(len(trials_blocks[i_block]), n_trials_per_block))
                else:
                    pass

        if np.sum(frequencies_combinations_repeated_and_split) != n_trials:

            raise Exception('At the end, np.sum(frequencies_combinations_repeated_and_split should be n_trials.\n'
                            'Now, np.sum(frequencies_combinations_repeated_and_split) = {}\n'
                            'and n_trials = {}.'.format(np.sum(frequencies_combinations_repeated_and_split),
                                                        n_trials))

        n_unused_trials = tmp_trials.shape[0]
        if n_unused_trials != 0:
            raise Exception(
                'n_unused_trials should be 0. From tmp_trials, n_unused_trials was {}.'.format(n_unused_trials))

    for i_block in range(n_blocks):
        trials_blocks[i_block] = cp_array.shuffle_in_windows(
            trials_blocks[i_block], n_samples_window=n_trials_per_shuffle,
            n_windows=None, axis=axis_combinations, dtype=None)

    return trials_blocks


def conditions_to_permutations(conditions, R, dtype=None):
    permutations = np.asarray(list(itertools.permutations(conditions, R)), dtype=dtype)
    return permutations


def conditions_to_P_random_permutetions(conditions, R=None, P=None, dtype=None):

    C = len(conditions)
    if R == C or R is None:
        n = math.factorial(C)
    else:
        n = cp_maths.convert_to_int_or_float(math.factorial(C) / math.factorial(C - R))

    if P is None or P >= n:
        permutations = conditions_to_permutations(conditions, R)
        permutations = cp_array.shuffle_in_any_dimension(permutations, 0)
        return permutations

    else:
        indexes_cond = np.arange(C)
        positions = np.arange(R)

        permutations_indexes = conditions_to_permutations(indexes_cond, R, dtype='i')
        permutations_indexes = cp_array.shuffle_in_any_dimension(permutations_indexes, 0)

        max_per_pos = math.ceil(20 / C)
        frequencies = np.empty([C, R], dtype='i')
        frequencies[:] = 0

        indexes_perm = np.empty(P, dtype='i')
        p = 0
        for i in range(n):
            if np.all(frequencies[tuple([indexes_cond[permutations_indexes[i, :]], positions])] < max_per_pos):
                frequencies[tuple([indexes_cond[permutations_indexes[i, :]], positions])] += 1
                indexes_perm[p] = p
                p += 1
                if p == P:
                    break

        if p == P:
            permutations_indexes = permutations_indexes[indexes_perm, :]
        elif 0 < p < P:
            indexes_perm = indexes_perm[:p]
            permutations_indexes = permutations_indexes[indexes_perm, :]
        else:
            permutations = None
            return permutations

        if np.equal(indexes_cond, conditions).all():
            return permutations_indexes
        else:
            if dtype is None:
                if isinstance(conditions, np.ndarray):
                    dtype = conditions.dtype
                else:
                    dtype = np.asarray(conditions).dtype
            permutations = np.empty([P, R], dtype=dtype)
            for c in range(C):
                permutations[permutations_indexes == c] = conditions[c]

            return permutations


