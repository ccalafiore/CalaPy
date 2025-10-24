import numpy as np
from copy import deepcopy
from . import descriptive
from .. import array as cp_array
from .. import maths as cp_maths
from .. import format as cp_format


def exclude_trials_with_rt_out_of_range(
        data, lower_limit, upper_limit, axes=None, indexes=None, value=np.nan, copy=False):
    
    
    if (axes is None) or (indexes is None):
        condition = np.logical_or(data < lower_limit, upper_limit < data)
    else:
        
        try:
            len(axes)
            axes_ = axes
        except TypeError:
            axes_ = [axes]
    
        try:
            len(indexes)
            indexes_ = indexes
        except TypeError:
            indexes_ = [indexes]
    
        condition = np.empty(data.shape, dtype=bool)
    
        n_axes = len(data.shape)
        indexes_data = np.empty(n_axes, dtype='O')
        indexes_condition = np.empty(n_axes, dtype='O')
        j = 0
        for a in range(n_axes):
            if a in axes_:
                try:
                    len(indexes_[j])
                except TypeError:
                    if not isinstance(indexes_[j], slice):
                        indexes_[j] = slice(indexes_[j], indexes_[j] + 1, 1)
    
                indexes_data[a] = indexes_[j]
                j += 1
            else:
                indexes_data[a] = slice(0, data.shape[a], 1)
    
            indexes_condition[a] = slice(0, data.shape[a], 1)
    
        data_to_check = data[tuple(indexes_data)]
    
        condition[tuple(indexes_condition)] = np.logical_or(data_to_check < lower_limit, upper_limit < data_to_check)

    if copy:
        data_out = deepcopy(data)
        data_out[condition] = value
        return data_out
    else:
        data[condition] = value
        return data


def exclude_subs_with_lower_accuracy(corrects, axis, threshold, order='n', ignore_nan=True):
    """

    Parameters
    ----------
    corrects
    axis
    threshold
    order: str, optional
        The desired outputs and their order. Accepted values are "n", "l", "a" or any combination of them like "al",
        "lan" (default is "n").
    ignore_nan

    Returns
    -------

    """
    n_outputs = len(order)
    outputs = [None] * n_outputs  # type: list

    for o in range(n_outputs):
        if order[o] in 'nla':

            accuracies_subs = descriptive.scores_to_means(corrects, axes=axis, ignore_nan=ignore_nan)

            if 'a' in order:
                outputs[order.index('a')] = accuracies_subs
            break

    for o in range(n_outputs):
        if order[o] in 'nl':
            subs_excluded_logical = accuracies_subs < threshold
            if 'l' in order:
                outputs[order.index('l')] = subs_excluded_logical
            break

    for o in range(n_outputs):
        if order[o] in 'n':
            subs_excluded_numeric = np.where(subs_excluded_logical)[0]
            if 'n' in order:
                outputs[order.index('n')] = subs_excluded_numeric
            break

    return outputs


def exclude_subs_with_lower_local_accuracy(corrects, axis, n_neighbours, threshold, order='n', ignore_nan=False):
    """

    Parameters
    ----------
    corrects
    axis
    n_neighbours
    threshold
    order: str, optional
        The desired outputs and their order. Accepted values are "n", "l", "m" or any combination of them like "ml",
        "lmn" (default is "n").
    ignore_nan

    Returns
    -------

    """
    n_outputs = len(order)
    outputs = [None] * n_outputs  # type: list

    for o in range(n_outputs):
        if order[o] in 'nlm':
            accuracy_local = descriptive.scores_to_local_mean(corrects, axis, n_neighbours, ignore_nan=ignore_nan)
            min_accuracy_local = accuracy_local.min(axis=axis, initial=1)
            if 'm' in order:
                outputs[order.index('m')] = min_accuracy_local
            break

    for o in range(n_outputs):
        if order[o] in 'nl':
            subs_excluded_logical = min_accuracy_local < threshold
            if 'l' in order:
                outputs[order.index('l')] = subs_excluded_logical
            break

    for o in range(n_outputs):
        if order[o] in 'n':
            subs_excluded_numeric = np.where(subs_excluded_logical)[0]
            if 'n' in order:
                outputs[order.index('n')] = subs_excluded_numeric
            break

    return outputs


def exclude_subs_with_higher_excluded_trials(data, axis, threshold, percentage=False, value=np.nan, order='n'):
    """

    Parameters
    ----------
    data : np.ndarray
        Some data with size S x N, where S is the number of subjects and N is the number of trials per subject.
        The excluded trials in the data were flagged with a specific value that you need to define in the
        parameter "value".
    axis : int or sequence or ints
        The axis of the data of the trials.
    threshold : int, float
        exclude subjects with excluded trials greater than threshold
    percentage : bool, optional
        If percentage is True, define threshold in percentage. Else, define threshold in number excluded trials.
    value : Any, optional
        The value that was used to flag the excluded trials in the data.
    order: str, optional
        The desired outputs and their order. Accepted values are "n", "l", "e" or any combination of them like "el",
        "len" (default is "n").

    Returns
    -------
        outputs : list
            A list of the elements requested in the parameter "order".

    """

    n_outputs = len(order)
    outputs = [None] * n_outputs  # type: list

    for o in range(n_outputs):
        if order[o] in 'nle':
            n_excluded_trials_subs = descriptive.scores_to_value_frequencies(data, axis, value)
            # if cp_maths.is_nan(value):
            #     excluded_trials_logical = cp_maths.is_nan(data)
            # else:
            #     excluded_trials_logical = data == value
            # n_excluded_trials_subs = np.sum(excluded_trials_logical, axis=axis, initial=0)
            break

    for o in range(n_outputs):
        if order[o] in 'nl':
            if percentage:
                N = data.shape[axis]

                if 'e' in order:
                    p_threshold = threshold
                    p_excluded_trials_subs = n_excluded_trials_subs * (100 / N)
                    subs_excluded_logical = p_excluded_trials_subs > p_threshold
                    outputs[order.index('e')] = p_excluded_trials_subs
                else:
                    n_threshold = threshold * (N / 100)
                    subs_excluded_logical = n_excluded_trials_subs > n_threshold
            else:
                n_threshold = threshold
                subs_excluded_logical = n_excluded_trials_subs > n_threshold

                if 'e' in order:
                    outputs[order.index('e')] = n_excluded_trials_subs

            if 'l' in order:
                outputs[order.index('l')] = subs_excluded_logical
            break

    for o in range(n_outputs):
        if order[o] in 'n':
            subs_excluded_numeric = np.where(subs_excluded_logical)[0]
            if 'n' in order:
                outputs[order.index('n')] = subs_excluded_numeric
            break

    return outputs


def exclude_subs_with_higher_local_excluded_trials(
        data, axis, n_neighbours, threshold, percentage=False, value=np.nan, order='n'):
    """

    Parameters
    ----------
    data : np.ndarray
        Some data with size S x N, where S is the number of subjects and N is the number of trials per subject.
        The excluded trials in the data were flagged with a specific value that you need to define in the
        parameter "value".
    axis : int or sequence or ints
        The axis of the data of the trials.
    threshold : int, float
        exclude subjects with excluded trials greater than threshold
    n_neighbours : int
        to add
    percentage : bool, optional
        If percentage is True, define threshold in percentage. Else, define threshold in number excluded trials.
    value : Any, optional
        The value that was used to flag the excluded trials in the data.
    order: str, optional
        The desired outputs and their order. Accepted values are "n", "l", "e" or any combination of them like "el",
        "len" (default is "n").

    Returns
    -------
        outputs : list
            A list of the elements requested in the parameter "order".

    """

    n_outputs = len(order)
    outputs = [None] * n_outputs  # type: list

    for o in range(n_outputs):
        if order[o] in 'nle':
            local_value_frequencies = descriptive.scores_to_local_value_frequencies(data, axis, n_neighbours, value)
            local_n_excluded_trials_subs = np.max(local_value_frequencies, axis=axis, initial=0)
            break

    for o in range(n_outputs):
        if order[o] in 'nl':
            if percentage:
                N = data.shape[axis]
                # N = n_neighbours

                if 'e' in order:
                    p_threshold = threshold
                    local_p_excluded_trials_subs = local_n_excluded_trials_subs * (100 / N)
                    subs_excluded_logical = local_p_excluded_trials_subs > p_threshold
                    outputs[order.index('e')] = local_p_excluded_trials_subs
                else:
                    n_threshold = threshold * (N / 100)
                    subs_excluded_logical = local_n_excluded_trials_subs > n_threshold
            else:
                n_threshold = threshold
                subs_excluded_logical = local_n_excluded_trials_subs > n_threshold

                if 'e' in order:
                    outputs[order.index('e')] = local_n_excluded_trials_subs

            if 'l' in order:
                outputs[order.index('l')] = subs_excluded_logical
            break

    for o in range(n_outputs):
        if order[o] in 'n':
            subs_excluded_numeric = np.where(subs_excluded_logical)[0]
            if 'n' in order:
                outputs[order.index('n')] = subs_excluded_numeric
            break

    return outputs


def exclude_subs(data, axis, subs_exc):

    if isinstance(data, (list, tuple)):
        data = np.asarray(data)

    subs_all = np.arange(data.shape[axis])
    subs_keep = subs_all[cp_array.samples_in_arr1_are_not_in_arr2(subs_all, subs_exc, axis=0)]

    n_axes = len(data.shape)
    indexes_data = np.empty(n_axes, dtype='O')
    for a in range(n_axes):
        if a == axis:
            indexes_data[a] = cp_format.numeric_indexes_to_slice(subs_keep)
        else:
            indexes_data[a] = slice(0, data.shape[a], 1)

    data_out = data[tuple(indexes_data)]

    return data_out
