import numpy as np
from scipy.stats import t
# from scipy.stats.stats import _ttest_finish
from ..combinations import n_conditions_to_combinations
from ..array import samples_in_arr1_are_in_arr2, advanced_indexing
from . import descriptive as cp_stats_descriptive


def scores_raw_to_scores_formatted(scores_raw, axis_comparisons, dtype=None):

    shape_scores_raw = np.asarray(scores_raw.shape)
    n_axes_scores_raw = len(shape_scores_raw)

    if axis_comparisons < 0:
        axis_comparisons += n_axes_scores_raw

    axis_comparisons_raw = axis_comparisons

    scores_raw = np.expand_dims(scores_raw, axis=axis_comparisons_raw)
    scores_raw = np.expand_dims(scores_raw, axis=axis_comparisons_raw)
    axis_comparisons_raw += 2

    n_conditions_iv_per_comparisons = 2

    shape_scores_raw = np.asarray(scores_raw.shape)
    n_axes_scores_raw = len(shape_scores_raw)
    n_conditions_iv = shape_scores_raw[axis_comparisons_raw]
    conditions_iv = np.arange(n_conditions_iv)

    axes_scores = np.arange(n_axes_scores_raw)
    axes_non_independent_variables_scores_raw = axes_scores[np.logical_not(samples_in_arr1_are_in_arr2(
        axes_scores, axis_comparisons_raw))]

    axes_independent_variable_formatted = np.asarray([
        axis_comparisons_raw - 2,
        axis_comparisons_raw - 1,
        axis_comparisons_raw], dtype=int)

    axes_non_independent_variable_formatted = axes_scores[np.logical_not(samples_in_arr1_are_in_arr2(
        axes_scores, axes_independent_variable_formatted))]

    shape_scores_formatted = np.copy(shape_scores_raw)
    shape_scores_formatted[axes_independent_variable_formatted[0]] = n_conditions_iv_per_comparisons
    shape_scores_formatted[axes_independent_variable_formatted[1:]] = n_conditions_iv

    if dtype is None:
        dtype = scores_raw.dtype
    # scores_formatted = np.full(shape_scores_formatted, 0, dtype=dtype)
    scores_formatted = np.empty(shape_scores_formatted, dtype=dtype)

    index_0_raw = np.empty(n_axes_scores_raw, dtype=object)
    index_0_formatted = np.copy(index_0_raw)

    for a in axes_non_independent_variables_scores_raw:
        index_0_raw[a] = np.arange(shape_scores_raw[a])

    index_0_formatted[axes_non_independent_variable_formatted] = index_0_raw[
        axes_non_independent_variable_formatted]
    # index_0_formatted[axes_independent_variable_formatted[0]] = np.arange(n_conditions_iv_per_comparisons)

    index_1_formatted = np.copy(index_0_formatted)

    index_0_formatted[axes_independent_variable_formatted[0]] = 0
    index_0_formatted[axes_independent_variable_formatted[2]] = conditions_iv

    index_1_formatted[axes_independent_variable_formatted[0]] = 1
    index_1_formatted[axes_independent_variable_formatted[1]] = conditions_iv

    for i_condition in range(n_conditions_iv):

        index_0_raw[axis_comparisons_raw] = i_condition
        index_0_raw_adv = advanced_indexing(index_0_raw)

        scores_raw_0 = scores_raw[index_0_raw_adv]

        index_0_formatted[axes_independent_variable_formatted[1]] = i_condition
        index_0_formatted_adv = advanced_indexing(index_0_formatted)

        index_1_formatted[axes_independent_variable_formatted[2]] = i_condition
        index_1_formatted_adv = advanced_indexing(index_1_formatted)

        scores_formatted[index_0_formatted_adv] = scores_raw_0

        scores_formatted[index_1_formatted_adv] = scores_raw_0
        # print(scores_raw_0.shape)
        # print(scores_formatted[index_0_formatted_adv].shape)
        # print(scores_formatted[index_1_formatted_adv].shape)

    return scores_formatted


def scores_raw_to_paired_t_test(
        scores_raw, axis_samples, axis_comparisons, alpha=0.05, tails='2', keepdims=False):

    np.seterr(divide='ignore', invalid='ignore')

    shape_scores_raw = scores_raw.shape
    n_axes_scores_raw = len(shape_scores_raw)
    if axis_samples < 0:
        axis_samples += n_axes_scores_raw
    if axis_comparisons < 0:
        axis_comparisons += n_axes_scores_raw

    axis_comparisons_raw = axis_comparisons
    axis_samples_raw = axis_samples
    scores_formatted = scores_raw_to_scores_formatted(
        scores_raw, axis_comparisons=axis_comparisons_raw)

    if axis_samples_raw > axis_comparisons_raw:
        axis_samples_formatted = axis_samples_raw + 2
    else:
        axis_samples_formatted = axis_samples_raw

    axis_comparisons_formatted = axis_comparisons

    diff = cp_stats_descriptive.scores_to_diff_of_scores(
        scores_formatted, axis=axis_comparisons_formatted, keepdims=keepdims)

    if keepdims:
        axis_samples_diff = axis_samples_formatted
    else:
        axis_samples_diff = axis_samples_formatted - (axis_samples_formatted > axis_comparisons_formatted)

    n_samples = np.sum(np.logical_not(np.isnan(diff)), axis=axis_samples_diff, keepdims=keepdims)
    means_of_diff = cp_stats_descriptive.scores_to_means(
        diff, axes=axis_samples_diff, keepdims=keepdims)
    std_err_of_diff = cp_stats_descriptive.scores_to_standard_errors(
        diff, axis_samples=axis_samples_diff, keepdims=keepdims)

    t_values = means_of_diff / std_err_of_diff
    np.nan_to_num(t_values, copy=False, nan=0.0)

    confidence = 1 - alpha
    df = n_samples - 1
    shape_df = df.shape
    # n_df = df.size
    indexes_df = n_conditions_to_combinations(shape_df)
    t_critical = np.empty(shape_df, dtype=float)
    p_values = np.empty(shape_df, dtype=float)

    for indexes_df_i in indexes_df:
        indexes_df_i_tuple = tuple(indexes_df_i)
        if tails == '2':
            t_critical[indexes_df_i_tuple] = t.ppf((1 + confidence) / 2., df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = (1.0 - t.cdf(abs(t_values[indexes_df_i_tuple]), df[indexes_df_i_tuple])) * 2
        elif tails == '1l':
            t_critical[indexes_df_i_tuple] = -t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])
        elif tails == '1r':
            t_critical[indexes_df_i_tuple] = t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = 1.0 - t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])

    # h = std_err_of_diff * t_critical

    return t_values, df, t_critical, p_values  # , h


def scores_formatted_to_paired_t_test(
        scores_formatted, axis_samples=-1, axis_comparisons=0, alpha=0.05, tails='2', keepdims=False):

    np.seterr(divide='ignore', invalid='ignore')

    shape_scores_formatted = scores_formatted.shape
    n_axes_scores_formatted = len(shape_scores_formatted)
    axis_samples_formatted = axis_samples % n_axes_scores_formatted
    axis_comparisons_formatted = axis_comparisons % n_axes_scores_formatted

    diff = cp_stats_descriptive.scores_to_diff_of_scores(
        scores_formatted, axis=axis_comparisons_formatted, keepdims=keepdims)

    if keepdims:
        axis_samples_diff = axis_samples_formatted
    else:
        axis_samples_diff = axis_samples_formatted - (axis_samples_formatted > axis_comparisons_formatted)

    n_samples = np.sum(np.logical_not(np.isnan(diff)), axis=axis_samples_diff, keepdims=keepdims)
    means_of_diff = cp_stats_descriptive.scores_to_means(
        diff, axes=axis_samples_diff, keepdims=keepdims)
    std_err_of_diff = cp_stats_descriptive.scores_to_standard_errors(
        diff, axis_samples=axis_samples_diff, keepdims=keepdims)

    t_values = means_of_diff / std_err_of_diff
    np.nan_to_num(t_values, copy=False, nan=0.0)

    confidence = 1 - alpha
    df = n_samples - 1
    shape_df = df.shape
    # n_df = df.size
    indexes_df = n_conditions_to_combinations(shape_df)
    t_critical = np.empty(shape_df, dtype=float)
    p_values = np.empty(shape_df, dtype=float)

    for indexes_df_i in indexes_df:
        indexes_df_i_tuple = tuple(indexes_df_i)
        if tails == '2':
            t_critical[indexes_df_i_tuple] = t.ppf((1 + confidence) / 2., df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = (1.0 - t.cdf(abs(t_values[indexes_df_i_tuple]), df[indexes_df_i_tuple])) * 2
        elif tails == '1l':
            t_critical[indexes_df_i_tuple] = -t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])
        elif tails == '1r':
            t_critical[indexes_df_i_tuple] = t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = 1.0 - t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])

    # h = std_err_of_diff * t_critical

    return t_values, df, t_critical, p_values  # , h


def diff_to_paired_t_test(
        diff, axis_samples=-1, alpha=0.05, tails='2', keepdims=False):

    np.seterr(divide='ignore', invalid='ignore')

    shape_diff = diff.shape
    n_axes_diff = len(shape_diff)
    axis_samples_diff = axis_samples % n_axes_diff

    n_samples = np.sum(np.logical_not(np.isnan(diff)), axis=axis_samples_diff, keepdims=keepdims)
    means_of_diff = cp_stats_descriptive.scores_to_means(
        diff, axes=axis_samples_diff, keepdims=keepdims)
    std_err_of_diff = cp_stats_descriptive.scores_to_standard_errors(
        diff, axis_samples=axis_samples_diff, keepdims=keepdims)

    t_values = means_of_diff / std_err_of_diff
    np.nan_to_num(t_values, copy=False, nan=0.0)

    confidence = 1 - alpha
    df = n_samples - 1
    shape_df = df.shape
    # n_df = df.size
    indexes_df = n_conditions_to_combinations(shape_df)
    t_critical = np.empty(shape_df, dtype=float)
    p_values = np.empty(shape_df, dtype=float)

    for indexes_df_i in indexes_df:
        indexes_df_i_tuple = tuple(indexes_df_i)
        if tails == '2':
            t_critical[indexes_df_i_tuple] = t.ppf((1 + confidence) / 2., df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = (1.0 - t.cdf(abs(t_values[indexes_df_i_tuple]), df[indexes_df_i_tuple])) * 2
        elif tails == '1l':
            t_critical[indexes_df_i_tuple] = -t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])
        elif tails == '1r':
            t_critical[indexes_df_i_tuple] = t.ppf(confidence, df[indexes_df_i_tuple])
            p_values[indexes_df_i_tuple] = 1.0 - t.cdf(t_values[indexes_df_i_tuple], df[indexes_df_i_tuple])

    # h = std_err_of_diff * t_critical

    return t_values, df, t_critical, p_values  # , h
