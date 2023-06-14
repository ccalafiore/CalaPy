import numpy as np
from . import format as cc_format


def values_are_not_repeated(dictionare, name_dictionary=None):
    keys, values = cc_format.dict_to_key_array_and_value_array(dictionare)
    for v in values:
        if (v == values).sum() > 1:
            keys_equal = keys[v == values]
            if name_dictionary is None:
                name_dictionary = 'dictionare'
            template_names_keys = name_dictionary + '[\'{}\']'
            raise ValueError(
                'The following assumption is not met:\n'
                '\t' + ' \u2260 '.join([template_names_keys.format(k) for k in keys_equal]))


def keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        dictionary, keys_necessary_known, values_necessary_known, name_dictionary=None):

    if not isinstance(dictionary, dict):
        if name_dictionary is None:
            raise TypeError('The type of dictionary must be dict')
        else:
            raise TypeError('The type of {} must be dict'.format(name_dictionary))

    if not isinstance(keys_necessary_known, np.ndarray):
        if isinstance(keys_necessary_known, list) or isinstance(keys_necessary_known, tuple):
            keys_necessary_known = np.asarray(keys_necessary_known, dtype=str)
        elif isinstance(keys_necessary_known, dict) or isinstance(keys_necessary_known, set):
            raise TypeError('The type of keys_necessary_known anything, but dict or set')
        else:
            keys_necessary_known = np.asarray([keys_necessary_known], dtype=str)

    if not isinstance(values_necessary_known, np.ndarray):
        if isinstance(values_necessary_known, list) or isinstance(values_necessary_known, tuple):
            values_necessary_known = np.asarray(values_necessary_known, dtype=str)
        elif isinstance(values_necessary_known, dict) or isinstance(values_necessary_known, set):
            raise TypeError('The type of values_necessary_known anything, but dict or set')
        else:
            values_necessary_known = np.asarray([values_necessary_known], dtype=str)

    if name_dictionary is None:
        message_1 = 'In the dictionary, '
    else:
        message_1 = 'In the dictionary ' + name_dictionary + ', '

    # string_keys_necessary_known = '\', \''.join(keys_necessary_known) + '.'
    message_unknown_keys_2 = 'the Key {} is unknown.\n'
    message_unknown_keys_3 = ('the following Keys are known:\n'
                              '\t{}.'.format(keys_necessary_known))
    message_unknown_keys = (message_1 + message_unknown_keys_2 +
                            message_1 + message_unknown_keys_3)

    message_missing_keys_2 = 'the following Keys are missing:\n\t{}.\n'
    message_missing_keys_3 = ('the following Keys are necessary:\n'
                              '\t{}.'.format(keys_necessary_known))
    message_missing_keys = (message_1 + message_missing_keys_2 +
                            message_1 + message_missing_keys_3)

    # string_values_necessary_known = '\', \''.join(values_necessary_known.astype(dtype=str)) + '.'
    message_unknown_values_2 = 'the Value {} is unknown.\n'
    message_unknown_values_3 = ('the following Values are known:\n'
                                '\t{}.'.format(values_necessary_known))
    message_unknown_values = (message_1 + message_unknown_values_2 +
                              message_1 + message_unknown_values_3)

    message_missing_values_2 = 'the following Values are missing:\n\t{}.\n'
    message_missing_values_3 = ('the following Values are necessary:\n'
                                '\t{}.'.format(values_necessary_known))
    message_missing_values = (message_1 + message_missing_values_2 +
                              message_1 + message_missing_values_3)
    # print('message_unknown_keys\n', message_unknown_keys)
    # print('message_missing_keys\n', message_missing_keys)
    # print('message_unknown_values\n', message_unknown_values)
    # print('message_missing_values:\n', message_missing_values)

    n_keys = len(keys_necessary_known)
    keys_missing = np.empty(n_keys, dtype=bool)
    keys_missing[:] = True

    n_values = len(values_necessary_known)
    values_missing = np.empty(n_values, dtype=bool)
    values_missing[:] = True
    for k, v in dictionary.items():

        if k in keys_necessary_known:
            keys_missing[k == keys_necessary_known] = False
        else:
            raise ValueError(message_unknown_keys.format(k))

        if v in values_necessary_known:
            values_missing[v == values_necessary_known] = False
        else:
            raise ValueError(message_unknown_values.format(v))

    if keys_missing.any():
        raise ValueError(message_missing_keys.format(keys_necessary_known[keys_missing]))
    if values_missing.any():
        raise ValueError(message_missing_values.format(values_necessary_known[values_missing]))
