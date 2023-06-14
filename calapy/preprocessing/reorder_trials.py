import numpy as np
from .. import combinations as cc_comb
from .. import array as cc_array
# TODO: remove cc_array.advanced_indexing


def reorder_trials(
        trials,
        axis_variables_table=-1,
        axis_samples=-2,
        independent_variables_table=0,
        dtype=None):

    if dtype is None:
        dtype = trials.dtype

    shape_trials = np.asarray(trials.shape, dtype=int)
    n_axes = len(shape_trials)
    axis_variables_table %= n_axes
    axis_samples %= n_axes
    n_trials_per_table = shape_trials[axis_samples]

    n_variables_table = shape_trials[axis_variables_table]
    variables_per_table = np.arange(n_variables_table)

    try:
        len(independent_variables_table)
    except TypeError:
        independent_variables_table = [independent_variables_table]
    independent_variables_table = np.asarray(independent_variables_table, dtype=int)
    n_independent_variables_table = len(independent_variables_table)

    axes = np.arange(n_axes)
    axes_table = np.sort([axis_samples, axis_variables_table])
    variables_axes = axes[np.logical_not(cc_array.samples_in_arr1_are_in_arr2(axes, axes_table))]
    variables_axes_inverted = variables_axes[::-1]

    shape_trials_ordered = shape_trials
    trials_ordered = np.empty(shape_trials_ordered, dtype=dtype)

    indexes_array_independent_variables = np.full(n_axes, 0, dtype=object)
    indexes_array_independent_variables[axis_variables_table] = independent_variables_table
    indexes_array_independent_variables[axis_samples] = np.arange(shape_trials[axis_samples])
    array_independent_variables_1_case = trials[cc_array.advanced_indexing(indexes_array_independent_variables)]

    for a in variables_axes_inverted:
        array_independent_variables_1_case = np.squeeze(array_independent_variables_1_case, axis=a)

    axis_variables_in_combinations = int(axis_variables_table > axis_samples)
    axis_combinations_in_combinations = int(not (bool(axis_variables_in_combinations)))

    conditions_independent_variables = cc_comb.trials_to_conditions(
        array_independent_variables_1_case, axis_combinations=axis_combinations_in_combinations)
    combinations_independent_variables = cc_comb.conditions_to_combinations(
        conditions_independent_variables,
        axis_combinations=axis_combinations_in_combinations)
    n_combinations_independent_variables = combinations_independent_variables.shape[axis_combinations_in_combinations]

    combinations_variables_axes = cc_comb.n_conditions_to_combinations(shape_trials[variables_axes])
    n_combinations_variables_axes = len(combinations_variables_axes)

    indexes_trials = np.empty(n_axes, dtype=object)
    indexes_trials[axis_variables_table] = variables_per_table

    indexes_trials_ordered = np.empty(n_axes, dtype=object)
    indexes_trials_ordered[axis_variables_table] = variables_per_table

    indexes_combinations_independent_variables = np.empty(2, dtype=object)
    indexes_combinations_independent_variables[axis_variables_in_combinations] = np.arange(n_independent_variables_table)

    if n_combinations_variables_axes == 0:
        t = 0
        for c in range(n_combinations_independent_variables):
            indexes_combinations_independent_variables[axis_combinations_in_combinations] = c

            indexes_trials[axis_samples] = np.all(
                array_independent_variables_1_case ==
                combinations_independent_variables[
                    cc_array.advanced_indexing(indexes_combinations_independent_variables)],
                axis=axis_variables_in_combinations)

            k = np.sum(indexes_trials[axis_samples])
            indexes_trials_ordered[axis_samples] = np.arange(t, t + k)
            trials_ordered[cc_array.advanced_indexing(indexes_trials_ordered)] = (
                trials[cc_array.advanced_indexing(indexes_trials)])
            t += k
    else:
        indexes_array_independent_variables = np.empty(n_axes, dtype=object)
        indexes_array_independent_variables[axis_samples] = np.arange(n_trials_per_table)
        indexes_array_independent_variables[axis_variables_table] = independent_variables_table
        for i in range(n_combinations_variables_axes):

            indexes_array_independent_variables[variables_axes] = combinations_variables_axes[i]

            array_independent_variables_i = trials[cc_array.advanced_indexing(indexes_array_independent_variables)]
            for a in variables_axes_inverted:
                array_independent_variables_i = np.squeeze(array_independent_variables_i, axis=a)

            indexes_trials[variables_axes] = combinations_variables_axes[i]
            indexes_trials_ordered[variables_axes] = combinations_variables_axes[i]

            t = 0
            for c in range(n_combinations_independent_variables):

                indexes_combinations_independent_variables[axis_combinations_in_combinations] = c

                indexes_trials[axis_samples] = np.all(
                    array_independent_variables_i ==
                    combinations_independent_variables[
                        cc_array.advanced_indexing(indexes_combinations_independent_variables)],
                    axis=axis_variables_in_combinations)

                k = np.sum(indexes_trials[axis_samples])
                indexes_trials_ordered[axis_samples] = np.arange(t, t + k)
                trials_ordered[cc_array.advanced_indexing(indexes_trials_ordered)] = (
                    trials[cc_array.advanced_indexing(indexes_trials)])
                t += k

    return trials_ordered



