# https://stackoverflow.com/questions/22611446/perform-2-sample-t-test

import numpy as np
import numpy.ma as ma
from scipy.stats import t
# from scipy.stats.stats import _ttest_finish
from . import descriptive as cc_stats_descriptive
from ..combinations import n_conditions_to_combinations


def scores_raw_to_scores_formatted(scores_raw, dtype=None):

    axis_object_independent_variable_raw = 0
    axis_object_comparisons_formatted = axis_object_independent_variable_raw
    axes_object_independent_variable_formatted = np.asarray([
        axis_object_comparisons_formatted + 1, axis_object_comparisons_formatted + 2], dtype=int)


    n_conditions_iv_per_comparisons = 2
    if isinstance(scores_raw, np.ndarray):

        n_conditions_iv = scores_raw.size
    else:
        n_conditions_iv = len(scores_raw)

    if dtype is not None:
        for c in range(n_conditions_iv):
            scores_raw[c] = scores_raw[c].astype(dtype, copy=False)

    n_axes_scores_formatted_object = 3
    shape_scores_formatted_object = np.empty(n_axes_scores_formatted_object, dtype=int)
    shape_scores_formatted_object[axis_object_comparisons_formatted] = n_conditions_iv_per_comparisons
    shape_scores_formatted_object[axes_object_independent_variable_formatted] = n_conditions_iv

    scores_formatted = np.empty(shape_scores_formatted_object, dtype=object)

    index_0_formatted_object = np.empty(n_axes_scores_formatted_object, dtype=object)
    index_0_formatted_object[:] = slice(None)
    index_1_formatted_object = np.copy(index_0_formatted_object)

    index_0_formatted_object[axis_object_comparisons_formatted] = 0
    index_1_formatted_object[axis_object_comparisons_formatted] = 1

    for i_condition in range(n_conditions_iv):

        scores_raw_0 = [scores_raw[i_condition]]

        index_0_formatted_object[axes_object_independent_variable_formatted[0]] = i_condition
        index_1_formatted_object[axes_object_independent_variable_formatted[1]] = i_condition

        scores_formatted[tuple(index_0_formatted_object)] = scores_raw_0
        scores_formatted[tuple(index_1_formatted_object)] = scores_raw_0
    return scores_formatted


def scores_formatted_to_df_in_unpaired_t_test(
        scores_formatted, axis_comparisons, axis_samples, equal_variances=False, keepdims=False, formt=True):

    if formt:
        shape_object_scores_formatted = np.asarray(scores_formatted.shape)
        n_axes_object_scores_formatted = shape_object_scores_formatted.size
        if axis_comparisons < 0:
            axis_comparisons += n_axes_object_scores_formatted

        indexes_object = np.empty(n_axes_object_scores_formatted, dtype=object)
        indexes_object[:] = 0
        shape_scores_formatted = np.asarray(scores_formatted[tuple(indexes_object)].shape)
        n_axes_scores_formatted = shape_scores_formatted.size
        if axis_samples < 0:
            axis_samples += n_axes_scores_formatted

    n_samples = cc_stats_descriptive.scores_to_n_samples(scores_formatted, axis_samples, keepdims=keepdims)

    shape_n_samples = np.asarray(n_samples.shape)
    n_axes_n_samples = shape_n_samples.size
    indexes_0 = np.empty(n_axes_n_samples, dtype=object)
    indexes_0[:] = slice(None)
    indexes_0[axis_comparisons] = 0

    indexes_1 = np.copy(indexes_0)
    indexes_1[axis_comparisons] = 1
    indexes_0_tuple = tuple(indexes_0)
    indexes_1_tuple = tuple(indexes_1)

    # EQUATIONS ARE AT https://www.statsdirect.co.uk/help/parametric_methods/utt.htm

    if equal_variances:
        df = n_samples[indexes_0_tuple] + n_samples[indexes_1_tuple] - 2
    else:
        variances = cc_stats_descriptive.scores_to_variances(scores_formatted, axis_samples, keepdims=keepdims)

        numerator_df = ((variances[indexes_0_tuple] / n_samples[indexes_0_tuple]) +
                        (variances[indexes_1_tuple] / n_samples[indexes_1_tuple]))**2

        denominator_df = ((((variances[indexes_0_tuple] / n_samples[indexes_0_tuple])**2) /
                           (n_samples[indexes_0_tuple] - 1)) +
                          (((variances[indexes_1_tuple] / n_samples[indexes_1_tuple]) ** 2) /
                           (n_samples[indexes_1_tuple] - 1)))

        df = numerator_df / denominator_df

    if keepdims:
        df = np.expand_dims(df, axis=axis_comparisons)

    return df


def scores_formatted_to_denominator_of_unpaired_t_test(
        scores_formatted, axis_comparisons, axis_samples, equal_variances=False, keepdims=False, formt=True):

    if formt:
        shape_object_scores_formatted = np.asarray(scores_formatted.shape)
        n_axes_object_scores_formatted = shape_object_scores_formatted.size
        if axis_comparisons < 0:
            axis_comparisons += n_axes_object_scores_formatted

        indexes_object = np.empty(n_axes_object_scores_formatted, dtype=object)
        indexes_object[:] = 0
        shape_scores_formatted = np.asarray(scores_formatted[tuple(indexes_object)].shape)
        n_axes_scores_formatted = shape_scores_formatted.size
        if axis_samples < 0:
            axis_samples += n_axes_scores_formatted

    n_samples = cc_stats_descriptive.scores_to_n_samples(scores_formatted, axis_samples, keepdims=keepdims)
    variances = cc_stats_descriptive.scores_to_variances(scores_formatted, axis_samples, keepdims=keepdims)

    shape_n_samples = np.asarray(n_samples.shape)
    n_axes_n_samples = shape_n_samples.size
    indexes_0 = np.empty(n_axes_n_samples, dtype=object)
    indexes_0[:] = slice(None)
    indexes_0[axis_comparisons] = 0

    indexes_1 = np.copy(indexes_0)
    indexes_1[axis_comparisons] = 1
    indexes_0_tuple = tuple(indexes_0)
    indexes_1_tuple = tuple(indexes_1)

    if equal_variances:
        # TO DO:
        # EQUATIONS ARE AT https://www.statsdirect.co.uk/help/parametric_methods/utt.htm
        df = ma.asanyarray(n_samples[indexes_0_tuple] + n_samples[indexes_1_tuple] - 2.0)
        svar = (
            (n_samples[indexes_0_tuple] - 1) * variances[indexes_0_tuple] +
            (n_samples[indexes_1_tuple] - 1) * variances[indexes_1_tuple]) / df

        denominator = ma.sqrt(svar * (1.0 / n_samples[indexes_0_tuple] + 1.0 / n_samples[indexes_1_tuple]))  # n-D computation here!
    else:
        denominator = np.sqrt(
            (variances[indexes_0_tuple] / n_samples[indexes_0_tuple]) +
            (variances[indexes_1_tuple] / n_samples[indexes_1_tuple]))

    if keepdims:
        denominator = np.expand_dims(denominator, axis=axis_comparisons)

    return denominator


def scores_formatted_to_unpaired_t_test(
        scores_formatted, axis_comparisons, axis_samples, equal_variances=False, alpha=0.05, tails='2', keepdims=False):

    np.seterr(divide='ignore', invalid='ignore')

    confidence = 1 - alpha

    if scores_formatted.dtype == object:
        shape_object_scores_formatted = np.asarray(scores_formatted.shape)
        n_axes_object_scores_formatted = shape_object_scores_formatted.size
        # axes_object_scores_formatted = np.arange(n_axes_object_scores_formatted)
        if axis_comparisons < 0:
            axis_comparisons += n_axes_object_scores_formatted

        indexes_object = np.empty(n_axes_object_scores_formatted, dtype=object)
        indexes_object[:] = 0
        shape_scores_formatted = np.asarray(scores_formatted[tuple(indexes_object)].shape)
        n_axes_scores_formatted = shape_scores_formatted.size
        if axis_samples < 0:
            axis_samples += n_axes_scores_formatted
    else:
        shape_scores_formatted = np.asarray(scores_formatted.shape)
        n_axes_scores_formatted = shape_scores_formatted.size
        if axis_comparisons < 0:
            axis_comparisons += n_axes_scores_formatted
        if axis_samples < 0:
            axis_samples += n_axes_scores_formatted

    means_of_scores = cc_stats_descriptive.scores_to_means(scores_formatted, axis_samples, keepdims=keepdims)
    diff_of_means = cc_stats_descriptive.scores_to_diff_of_scores(means_of_scores, axis_comparisons, keepdims=keepdims)

    denominator = scores_formatted_to_denominator_of_unpaired_t_test(
        scores_formatted, axis_comparisons, axis_samples,
        equal_variances=equal_variances, keepdims=keepdims, formt=False)

    # t_values = diff_of_means / denominator
    t_values = np.asarray(diff_of_means / denominator)
    # np.nan_to_num(t_values, copy=False, nan=0.0)

    df = scores_formatted_to_df_in_unpaired_t_test(
        scores_formatted, axis_comparisons, axis_samples,
        equal_variances=equal_variances, keepdims=keepdims, formt=False)

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

    return t_values, df, t_critical, p_values


# def height_of_t_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), stats.sem(a)
#     h = se * stats.t.ppf((1 + confidence) / 2., n-1)
#     return h
