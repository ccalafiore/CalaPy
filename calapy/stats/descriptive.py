import math

import numpy as np
import torch
from scipy.stats import t
# from scipy.stats.stats import _ttest_finish
from .. import combinations as cp_combinations
from .. import array as cp_array
from .. import maths as cp_maths


def select_sum(ignore_nan=True):
    if ignore_nan:
        sum = np.nansum
    else:
        sum = np.sum
    return sum


def counter_excluding_nan_keeping_dims(scores, axis_samples):
    n_samples = np.sum(cp_maths.is_not_nan(scores), axis=axis_samples, keepdims=True)
    return n_samples


def counter_excluding_nan_not_keeping_dims(scores, axis_samples):
    n_samples = np.sum(cp_maths.is_not_nan(scores), axis=axis_samples, keepdims=False)
    return n_samples


# def counter_including_nan_keeping_dims(scores, axis_samples):
#     n_samples = np.empty(scores.shape, dtype='i')
#     n_samples[:] = scores.shape[axis_samples]
#     return n_samples
#
#
# def counter_including_nan_not_keeping_dims(scores, axis_samples):
#     n_samples = scores.shape[axis_samples]
#     return n_samples
def counter_including_nan(scores, axis_samples):
    n_samples = scores.shape[axis_samples]
    return n_samples


def select_counter(ignore_nan=True, keepdims=False):
    if ignore_nan:
        if keepdims:
            counter = counter_excluding_nan_keeping_dims
        else:
            counter = counter_excluding_nan_not_keeping_dims
    else:
        counter = counter_including_nan
        # if keepdims:
        #     counter = counter_including_nan_keeping_dims
        # else:
        #     counter = counter_including_nan_not_keeping_dims
    return counter


def scores_to_diff_of_scores(scores, axis, delta=1, stride=1, keepdims=False):
    # this function will go to cp.array in future versions

    shape_scores = np.asarray(scores.shape)
    n_axes_scores = shape_scores.size
    if axis < 0:
        axis += n_axes_scores

    n_conditions = shape_scores[axis]
    index_0 = np.empty(n_axes_scores, dtype='O')
    index_0[:] = slice(None)
    index_1 = np.copy(index_0)

    index_diff = np.copy(index_0)
    index_diff[axis] = 0

    n_differences = int((n_conditions - abs(delta)) // abs(stride))
    shape_diff_of_scores = shape_scores
    shape_diff_of_scores[axis] = n_differences
    diff_of_scores = np.empty(shape_diff_of_scores, dtype=scores.dtype)

    if delta > 0 and stride > 0:
        for i in range(0, n_conditions - delta, stride):
            index_0[axis] = i
            index_1[axis] = i + delta
            diff_of_scores[tuple(index_diff)] = scores[tuple(index_0)] - scores[tuple(index_1)]
            index_diff[axis] += 1
    elif delta < 0 and stride > 0:
        for i in range(abs(delta), n_conditions, stride):
            index_0[axis] = i
            index_1[axis] = i + delta
            diff_of_scores[tuple(index_diff)] = scores[tuple(index_0)] - scores[tuple(index_1)]
            index_diff[axis] += 1
    elif delta > 0 and stride < 0:
        for i in range(n_conditions - delta - 1, -1, stride):
            index_0[axis] = i
            index_1[axis] = i + delta
            diff_of_scores[tuple(index_diff)] = scores[tuple(index_0)] - scores[tuple(index_1)]
            index_diff[axis] += 1
    elif delta < 0 and stride < 0:
        for i in range(n_conditions - 1, abs(delta) -1, stride):
            index_0[axis] = i
            index_1[axis] = i + delta
            diff_of_scores[tuple(index_diff)] = scores[tuple(index_0)] - scores[tuple(index_1)]
            index_diff[axis] += 1
    elif delta == 0:
        raise ValueError('delta has to be an intiger smaller or greater than 0. It cannot be 0')
    elif stride == 0:
        raise ValueError('stride has to be an intiger smaller or greater than 0. It cannot be 0')
    else:
        raise ValueError('Both delta and stride have to be an intiger smaller or greater than 0.')

    if not keepdims and (diff_of_scores.shape[axis] == 1):
        diff_of_scores = np.squeeze(diff_of_scores, axis=axis)

    return diff_of_scores


def scores_to_n_samples(scores, axis_samples, keepdims=False, ignore_nan=True):

    counter = select_counter(ignore_nan=ignore_nan, keepdims=keepdims)

    if scores.dtype == object:

        shape_object_scores = np.asarray(scores.shape)
        n_axes_object_scores = shape_object_scores.size
        axes_object_scores = np.arange(n_axes_object_scores)

        indexes_object = np.empty(n_axes_object_scores, dtype='O')
        indexes_object[:] = 0
        shape_scores = np.asarray(scores[tuple(indexes_object)].shape)
        n_axes_scores = shape_scores.size
        if axis_samples < 0:
            axis_samples += n_axes_scores

        if keepdims:
            shape_scores_tmp = shape_scores
            shape_scores_tmp[axis_samples] = 1
        else:
            shape_scores_tmp = shape_scores[np.arange(n_axes_scores) != axis_samples]

        shape_n_samples = np.append(shape_object_scores, shape_scores_tmp)
        n_samples = np.empty(shape_n_samples, dtype='i')

        n_axes_n_samples = shape_n_samples.size
        indexes_n_samples = np.empty(n_axes_n_samples, dtype='O')
        indexes_n_samples[:] = slice(None)

        indexes_object = cp_combinations.n_conditions_to_combinations(shape_object_scores)
        for indexes_object_i in indexes_object:

            indexes_n_samples[axes_object_scores] = indexes_object_i
            indexes_n_samples_tuple = tuple(indexes_n_samples)
            indexes_object_i_tuple = tuple(indexes_object_i)

            n_samples[indexes_n_samples_tuple] = counter(scores[indexes_object_i_tuple], axis_samples=axis_samples)

            # n_samples[indexes_n_samples_tuple] = np.sum(
            #     cp_maths.is_not_nan(scores[indexes_object_i_tuple]), axis=axis_samples, keepdims=keepdims)

    else:
        shape_scores = np.asarray(scores.shape)
        n_axes_scores = shape_scores.size
        if axis_samples < 0:
            axis_samples += n_axes_scores

        n_samples = counter(scores, axis_samples=axis_samples)

    return n_samples


def samples_to_frequencies(data, axis_samples, axis_variables_table, exclude_values=False, values=-1):

    shape_data = np.asarray(data.shape, dtype='i')
    n_axes_data = shape_data.size

    if axis_samples < 0:
        axis_samples += n_axes_data
    if axis_variables_table < 0:
        axis_variables_table += n_axes_data

    if axis_samples < axis_variables_table:
        axis_comb = 0
    elif axis_samples > axis_variables_table:
        axis_comb = 1
    else:
        raise ValueError('The following assumption is not met:\n'
                         '\taxis_samples \u003D axis_variables_table')

    axes_data = np.arange(n_axes_data)
    axes_not_variables_data = axes_data[axes_data != axis_variables_table]
    axes_other_data = axes_not_variables_data[axes_not_variables_data != axis_samples]
    n_axes_other_data = axes_other_data.size
    n_variables_table = shape_data[axis_variables_table]

    indexes_data = np.empty(n_axes_data, dtype='O')
    indexes_data[:] = slice(None)

    # conditions = [None] * n_variables_table  # type: list
    conditions = None  # type: # list

    for indexes_axes_other in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_data[axes_other_data]):

        indexes_data[axes_other_data] = indexes_axes_other

        conditions_tmp = cp_combinations.trials_to_conditions(
            data[tuple(indexes_data)], axis_combinations=axis_comb, exclude_values=exclude_values, values=values)

        if conditions is None:
            conditions = conditions_tmp
        else:
            conditions = [np.unique(np.append(conditions[v], conditions_tmp[v])) for v in range(0, n_variables_table, 1)]

    n_conditions = cp_combinations.conditions_to_n_conditions(conditions)

    n_axes_frequencies = n_axes_other_data + n_variables_table
    shape_frequencies = np.empty(n_axes_frequencies, dtype='i')
    shape_frequencies[slice(n_axes_other_data)] = shape_data[axes_other_data]
    shape_frequencies[slice(n_axes_other_data, n_axes_frequencies)] = n_conditions
    frequencies = np.full(shape_frequencies, fill_value=0, dtype='i')
    indexes_frequencies = np.empty(n_axes_frequencies, dtype='O')

    combinations_not_variables = cp_combinations.n_conditions_to_combinations(shape_data[axes_not_variables_data])
    # combinations_others = combinations_not_variables[slice(None), axes_not_variables_data != axis_samples]
    indexes_comb_c = axes_not_variables_data != axis_samples
    for comb_c in combinations_not_variables:

        indexes_data[axes_not_variables_data] = comb_c
        data_c = data[tuple(indexes_data)]
        if exclude_values:
            if all(cp_array.samples_in_arr1_are_not_in_arr2(data_c, values)):
                indexes_frequencies[slice(n_axes_other_data)] = comb_c[indexes_comb_c]
                indexes_frequencies[slice(n_axes_other_data, n_axes_frequencies)] = data_c
                frequencies[tuple(indexes_frequencies)] += 1
        else:
            indexes_frequencies[slice(n_axes_other_data)] = comb_c[indexes_comb_c]
            indexes_frequencies[slice(n_axes_other_data, n_axes_frequencies)] = data_c
            frequencies[tuple(indexes_frequencies)] += 1

    return frequencies


def samples_to_local_frequencies(
        data, axis_samples, axis_variables_table, size_local, stride=1, exclude_values=False, values=-1):

    shape_data = np.asarray(data.shape, dtype='i')
    n_axes_data = shape_data.size

    if axis_samples < 0:
        axis_samples += n_axes_data
    if axis_variables_table < 0:
        axis_variables_table += n_axes_data

    if axis_samples < axis_variables_table:
        axis_comb = 0
    elif axis_samples > axis_variables_table:
        axis_comb = 1
    else:
        raise ValueError('The following assumption is not met:\n'
                         '\taxis_samples \u003D axis_variables_table')

    axes_data = np.arange(n_axes_data, dtype='i')
    axes_not_variables_data = axes_data[axes_data != axis_variables_table]
    axes_other_data = axes_not_variables_data[axes_not_variables_data != axis_samples]
    n_axes_other_data = axes_other_data.size
    n_axes_not_variables_data = axes_not_variables_data.size
    n_variables_table = shape_data[axis_variables_table]
    I_in = data.shape[axis_samples]

    indexes_data = np.asarray([slice(0, data.shape[a], 1) for a in range(0, n_axes_data, 1)], dtype='O')

    # conditions = [None] * n_variables_table  # type: list
    conditions = None  # type: # list

    for indexes_axes_other in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_data[axes_other_data]):

        indexes_data[axes_other_data] = indexes_axes_other

        conditions_tmp = cp_combinations.trials_to_conditions(
            data[tuple(indexes_data)], axis_combinations=axis_comb, exclude_values=exclude_values, values=values)

        if conditions is None:
            conditions = conditions_tmp
        else:
            conditions = [np.unique(np.append(conditions[v], conditions_tmp[v])) for v in
                          range(0, n_variables_table, 1)]

    n_conditions = cp_combinations.conditions_to_n_conditions(conditions)

    I_out = math.floor((I_in - size_local) / stride) + 1
    n_axes_freq = n_axes_not_variables_data + n_variables_table

    axes_freq = np.arange(n_axes_freq, dtype='i')

    if axis_samples < axis_variables_table:
        axis_samples_freq = axis_samples
    elif axis_samples > axis_variables_table:
        axis_samples_freq = axis_samples - 1
    else:
        raise ValueError('axis_samples == axis_variables_table')

    n_axes_not_variables_freq = n_axes_not_variables_data
    axes_not_variables_freq = np.arange(n_axes_not_variables_freq, dtype='i')
    axes_other_freq = axes_not_variables_freq[axes_not_variables_freq != axis_samples_freq]

    shape_frequencies = np.empty(n_axes_freq, dtype='i')
    shape_frequencies[axes_other_freq] = shape_data[axes_other_data]
    shape_frequencies[axis_samples_freq] = I_out
    shape_frequencies[slice(n_axes_not_variables_freq, n_axes_freq)] = n_conditions
    frequencies = np.full(shape_frequencies, fill_value=0, dtype='i')
    indexes_frequencies = np.empty(n_axes_freq, dtype='O')

    indexes_data = np.asarray([slice(0, data.shape[a], 1) for a in range(0, n_axes_data, 1)], dtype='O')
    indexes_data_i = np.asarray(
        [slice(0, data.shape[a], 1) if a != axis_samples else slice(0, size_local, 1)
         for a in range(0, n_axes_data, 1)], dtype='O')

    indexes_comb_c = axes_not_variables_data != axis_samples

    for i, start_i in enumerate(range(0, I_in - (size_local - 1), stride)):

        end_i = start_i + size_local
        indexes_data[axis_samples] = slice(start_i, end_i, 1)

        data_i = data[tuple(indexes_data)]

        shape_data_i = np.asarray(data_i.shape, dtype='i')

        indexes_frequencies[axis_samples_freq] = i

        for comb_c in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_data_i[axes_not_variables_data]):

            indexes_data_i[axes_not_variables_data] = comb_c
            data_ic = data_i[tuple(indexes_data_i)]
            if exclude_values and any(cp_array.samples_in_arr1_are_in_arr2(data_ic, values)):
                continue

            indexes_frequencies[axes_other_freq] = comb_c[indexes_comb_c]
            indexes_frequencies[slice(n_axes_not_variables_data, n_axes_freq)] = data_ic
            frequencies[tuple(indexes_frequencies)] += 1

    return frequencies


def samples_to_probabilities(data, axis_samples, axis_variables_table):

    if axis_samples == axis_variables_table:
        raise ValueError('The following assumption is not met:\n'
                         '\taxis_samples \u2260 axis_variables_table')
    n_samples = data.shape[axis_samples]
    frequencies = samples_to_frequencies(data, axis_samples, axis_variables_table)
    probabilities = frequencies / n_samples

    return probabilities


def samples_to_percentages(data, axis_samples, axis_variables_table):
    if axis_samples == axis_variables_table:
        raise ValueError('The following assumption is not met:\n'
                         '\taxis_samples \u2260 axis_variables_table')
    n_samples = data.shape[axis_samples]
    frequencies = samples_to_frequencies(data, axis_samples, axis_variables_table)
    percentages = frequencies / n_samples * 100

    return percentages


def scores_to_df_of_variance_and_paired_t(scores, axis_samples, keepdims=False, ignore_nan=True):

    n_samples = scores_to_n_samples(scores, axis_samples=axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)
    df = n_samples - 1

    return df


def scores_to_means(scores, axes, keepdims=False, ignore_nan=True):

    axes_means = axes

    sum = select_sum(ignore_nan=ignore_nan)

    if scores.dtype == object:

        shape_object_scores = np.asarray(scores.shape)
        n_axes_object_scores = shape_object_scores.size
        axes_object_scores = np.arange(n_axes_object_scores)

        indexes_object = np.empty(n_axes_object_scores, dtype='O')
        indexes_object[:] = 0
        shape_scores = np.asarray(scores[tuple(indexes_object)].shape, dtype='i')
        n_axes_scores = shape_scores.size
        try:
            len(axes_means)
            # n_axes_means = len(axes_means)
            axes_means = np.asarray(axes_means, dtype='i')
            axes_means[axes_means < 0] += n_axes_scores
            # check point 1
            if np.sum(axes_means[0] == axes_means) > 1:
                raise ValueError('axes cannot contain repeated values')
            axes_means = np.sort(axes_means)[::-1]
        except TypeError:
            if axes_means < 0:
                axes_means += n_axes_scores
            axes_means = np.asarray([axes_means], dtype='i')
            # n_axes_means = 1

        if keepdims:
            shape_scores_tmp = shape_scores
            shape_scores_tmp[axes_means] = 1
        else:
            axes_scores = np.arange(n_axes_scores)
            shape_scores_tmp = shape_scores[np.logical_not(cp_array.samples_in_arr1_are_in_arr2(
                axes_scores, axes_means))]

        shape_means = np.append(shape_object_scores, shape_scores_tmp)
        means = np.empty(shape_means, dtype='f')

        n_axes_means = shape_means.size
        indexes_means = np.empty(n_axes_means, dtype='O')
        indexes_means[:] = slice(None)

        indexes_object = cp_combinations.n_conditions_to_combinations(shape_object_scores)
        for indexes_object_i in indexes_object:

            indexes_means[axes_object_scores] = indexes_object_i
            indexes_means_tuple_i = tuple(indexes_means)
            indexes_object_tuple_i = tuple(indexes_object_i)

            scores_tmp_i = scores[indexes_object_tuple_i]
            for a in axes_means:
                n_samples = scores_to_n_samples(
                    scores_tmp_i, axis_samples=a, keepdims=keepdims, ignore_nan=ignore_nan)
                # n_samples = np.sum(cp_maths.is_not_nan(scores_tmp_i), axis=a, keepdims=keepdims)

                sum_of_scores = sum(scores_tmp_i, axis=a, keepdims=keepdims)
                scores_tmp_i = sum_of_scores / n_samples

            means[indexes_means_tuple_i] = scores_tmp_i

    else:

        shape_scores = np.asarray(scores.shape, dtype='i')
        n_axes_scores = shape_scores.size

        try:
            len(axes_means)
            # n_axes_means = len(axes_means)
            axes_means = np.asarray(axes_means, dtype='i')
            axes_means[axes_means < 0] += n_axes_scores
            # check point 1
            if np.sum(axes_means[0] == axes_means) > 1:
                raise ValueError('axes cannot contain repeated values')
            axes_means = np.sort(axes_means)[::-1]
        except TypeError:
            if axes_means < 0:
                axes_means += n_axes_scores
            axes_means = np.asarray([axes_means], dtype='i')
            # n_axes_means = 1
        scores_tmp = scores
        for a in axes_means:
            n_samples = scores_to_n_samples(
                scores_tmp, axis_samples=a, keepdims=keepdims, ignore_nan=ignore_nan)
            # n_samples = np.sum(cp_maths.is_not_nan(scores_tmp), axis=a, keepdims=keepdims)
            sum_of_scores = sum(scores_tmp, axis=a, keepdims=keepdims)
            scores_tmp = sum_of_scores / n_samples
        means = scores_tmp

    return means


def scores_to_variances(scores, axis_samples, keepdims=False, ignore_nan=True):

    sum = select_sum(ignore_nan=ignore_nan)

    if scores.dtype == object:

        shape_object_scores = np.asarray(scores.shape)
        n_axes_object_scores = shape_object_scores.size
        axes_object_scores = np.arange(n_axes_object_scores)

        indexes_object = np.empty(n_axes_object_scores, dtype='O')
        indexes_object[:] = 0
        shape_scores = np.asarray(scores[tuple(indexes_object)].shape)
        n_axes_scores = shape_scores.size
        if axis_samples < 0:
            axis_samples += n_axes_scores

        if keepdims:
            shape_scores_tmp = shape_scores
            shape_scores_tmp[axis_samples] = 1
        else:
            shape_scores_tmp = shape_scores[np.arange(n_axes_scores) != axis_samples]

        shape_variances = np.append(shape_object_scores, shape_scores_tmp)
        variances = np.empty(shape_variances, dtype='f')

        n_axes_variances = shape_variances.size
        indexes_variances = np.empty(n_axes_variances, dtype='O')
        indexes_variances[:] = slice(None)

        indexes_object = cp_combinations.n_conditions_to_combinations(shape_object_scores)
        for indexes_object_i in indexes_object:

            indexes_variances[axes_object_scores] = indexes_object_i
            indexes_variances_tuple = tuple(indexes_variances)
            indexes_object_i_tuple = tuple(indexes_object_i)

            n_samples = scores_to_n_samples(
                scores[indexes_object_i_tuple], axis_samples=axis_samples, keepdims=True, ignore_nan=ignore_nan)
            # n_samples = np.sum(cp_maths.is_not_nan(
            #     scores[indexes_object_i_tuple]), axis=axis_samples, keepdims=True)

            sum_of_scores = sum(scores[indexes_object_i_tuple], axis=axis_samples, keepdims=True)
            # sum_of_scores = np.nansum(scores[indexes_object_i_tuple], axis=axis_samples, keepdims=keepdims)

            means = sum_of_scores / n_samples
            # means = scores_to_means(
            #     scores[indexes_object_i_tuple], axes=axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)

            sum_of_squared_distances = sum(
                (scores[indexes_object_i_tuple] - means) ** 2, axis=axis_samples, keepdims=keepdims)

            if not keepdims:
                if isinstance(n_samples, (np.ndarray, torch.Tensor)):
                    n_samples.shape[axis_samples]
                    n_samples = np.squeeze(n_samples, axis=axis_samples)

            df = n_samples - 1

            variances[indexes_variances_tuple] = sum_of_squared_distances / df

    else:

        shape_scores = np.asarray(scores.shape)
        n_axes_scores = shape_scores.size
        if axis_samples < 0:
            axis_samples += n_axes_scores

        n_samples = scores_to_n_samples(scores, axis_samples=axis_samples, keepdims=True, ignore_nan=ignore_nan)
        # n_samples = np.sum(cp_maths.is_not_nan(scores), axis=axis_samples, keepdims=True)
        sum_of_scores = sum(scores, axis=axis_samples, keepdims=True)
        means = sum_of_scores / n_samples
        # means = np.nansum(scores, axis=axis_samples, keepdims=True) / n_samples
        sum_of_squared_distances = sum((scores - means) ** 2, axis=axis_samples, keepdims=keepdims)
        # sum_of_squared_distances = np.nansum((scores - means) ** 2, axis=axis_samples, keepdims=keepdims)

        if not keepdims:
            if isinstance(n_samples, (np.ndarray, torch.Tensor)):
                n_samples.shape[axis_samples]
                n_samples = np.squeeze(n_samples, axis=axis_samples)

        df = n_samples - 1
        variances = sum_of_squared_distances / df

    return variances


def scores_to_standard_deviations(scores, axis_samples, keepdims=False, ignore_nan=True):

    variances = scores_to_variances(scores, axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)
    standard_deviations = np.sqrt(variances)

    return standard_deviations


def scores_to_standard_errors(scores, axis_samples, keepdims=False, ignore_nan=True):

    # shape_scores = np.asarray(scores.shape)
    # n_axes_scores = shape_scores.size
    # if axis_samples < 0:
    #     axis_samples += n_axes_scores

    n_samples = scores_to_n_samples(scores, axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)
    variances = scores_to_variances(scores, axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)
    std_error = np.sqrt(variances / n_samples)

    return std_error


def scores_to_confidence_intervals(
        scores, axis_samples, alpha=0.05, tails='2', keepdims=False, ignore_nan=True):

    confidence = 1 - alpha

    # shape_scores = np.asarray(scores.shape)
    # n_axes_scores = shape_scores.size
    # if axis_samples < 0:
    #     axis_samples += n_axes_scores

    std_err = scores_to_standard_errors(scores, axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)
    df = scores_to_df_of_variance_and_paired_t(scores, axis_samples, keepdims=keepdims, ignore_nan=ignore_nan)

    if isinstance(df, (int, float)):

        if tails == '2':
            t_critical = t.ppf((1 + confidence) / 2., df)
        elif tails == '1l':
            t_critical = -t.ppf(confidence, df)
        elif tails == '1r':
            t_critical = t.ppf(confidence, df)
        else:
            raise TypeError('tails')

    elif isinstance(df, (np.ndarray, torch.Tensor)):

        shape_df = np.asarray(df.shape)
        indexes_df = cp_combinations.n_conditions_to_combinations(shape_df)
        t_critical = np.empty(shape_df, dtype='f')

        for indexes_df_i in indexes_df:

            indexes_df_i_tuple = tuple(indexes_df_i)

            if tails == '2':
                t_critical[indexes_df_i_tuple] = t.ppf((1 + confidence) / 2., df[indexes_df_i_tuple])
            elif tails == '1l':
                t_critical[indexes_df_i_tuple] = -t.ppf(confidence, df[indexes_df_i_tuple])
            elif tails == '1r':
                t_critical[indexes_df_i_tuple] = t.ppf(confidence, df[indexes_df_i_tuple])
            else:
                raise TypeError('tails')
    else:
        raise TypeError('df')

    h = std_err * t_critical

    return h


def scores_to_local_mean(data, axis, size_local, stride=1, ignore_nan=True):

    n_axes = len(data.shape)
    if axis < 0:
        axis += n_axes

    I_in = data.shape[axis]
    I_out = math.floor((I_in - size_local) / stride) + 1

    shape_means_local = list(data.shape)

    shape_means_local[axis] = I_out

    means_local = np.empty(shape_means_local, dtype='f')

    indexes_means_local = np.empty(n_axes, dtype='O')
    indexes_data = np.empty(n_axes, dtype='O')

    for a in range(n_axes):
        if a != axis:
            indexes_means_local[a] = slice(0, means_local.shape[a], 1)
            indexes_data[a] = slice(0, data.shape[a], 1)

    for i, start_i in enumerate(range(0, I_in - (size_local - 1), stride)):

        end_i = start_i + size_local
        indexes_data[axis] = slice(start_i, end_i, 1)

        indexes_means_local[axis] = i

        means_local[tuple(indexes_means_local)] = scores_to_means(
            data[tuple(indexes_data)], axes=axis, keepdims=False, ignore_nan=ignore_nan)

    return means_local


def scores_to_value_frequencies(data, axis, value):

    if cp_maths.is_nan(value):
        arr_logical = cp_maths.is_nan(data)
    else:
        arr_logical = data == value

    arr = np.sum(arr_logical, axis=axis, initial=0)
    return arr


def scores_to_local_value_frequencies(data, axis, n_neighbours, value):
    n_axes = len(data.shape)
    if axis < 0:
        axis += n_axes

    I = data.shape[axis]

    shape_local_frequencies = list(data.shape)
    shape_local_frequencies[axis] = I - n_neighbours + 1

    local_frequencies = np.empty(shape_local_frequencies, dtype='i')

    indexes_local_frequencies = np.empty(n_axes, dtype='O')
    indexes_data = np.empty(n_axes, dtype='O')

    for a in range(n_axes):
        if a != axis:
            indexes_local_frequencies[a] = slice(0, local_frequencies.shape[a], 1)
            indexes_data[a] = slice(0, data.shape[a], 1)

    for i in range(0, I - n_neighbours + 1, 1):
        indexes_local_frequencies[axis] = i
        indexes_data[axis] = slice(i, i + n_neighbours, 1)

        local_frequencies[tuple(indexes_local_frequencies)] = scores_to_value_frequencies(
            data[tuple(indexes_data)], axis, value)

    return local_frequencies
