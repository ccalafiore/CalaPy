import numpy as np


def numeric_indexes_to_slice(indexes):

    if not isinstance(indexes, slice):
        try:
            n_indexes_a = len(indexes)
            if n_indexes_a == 0:
                pass
            elif n_indexes_a == 1:
                # indexes = slice(indexes[0], indexes[0] + 1, 1)
                indexes = indexes[0]
            elif n_indexes_a > 1:
                sliceable = True
                step = indexes[1] - indexes[0]
                for i in range(2, n_indexes_a):
                    step_i = indexes[i] - indexes[i - 1]
                    if step_i != step:
                        sliceable = False
                        break
                if sliceable:
                    if step > 0:
                        stop = indexes[-1] + 1
                    else:
                        stop = indexes[-1] - 1
                        if stop < 0:
                            stop = None
                    indexes = slice(indexes[0], stop, step)

        except TypeError:
            pass

    return indexes


def seq_of_numeric_indexes_to_seq_of_slices(indexes):
    n_axes = len(indexes)
    for a in range(n_axes):
        indexes[a] = numeric_indexes_to_slice(indexes[a])
    return indexes


def format_shape_arguments(dict_arguments, target_shape):

    if isinstance(target_shape, list) or isinstance(target_shape, tuple):
        target_shape = np.asarray(target_shape, dtype=int)
    if isinstance(target_shape, np.ndarray):
        pass
    else:
        target_shape = np.asarray([target_shape], dtype=int)

    n_axes = target_shape.size

    # n_arguments = len(arguments)
    for key_i, argument_i in dict_arguments.items():

        # format arguments[i]
        if argument_i is None:
            argument_i = np.empty(target_shape, dtype=object)

        elif isinstance(argument_i, str):

            len_argument_i = len(argument_i)
            dtype_argument_i = '<U' + str(len_argument_i)
            argument_i_tmp = argument_i
            argument_i = np.empty(target_shape, dtype=dtype_argument_i)
            argument_i[:] = argument_i_tmp

        elif (isinstance(argument_i, list) or
              isinstance(argument_i, np.ndarray) or
              isinstance(argument_i, tuple)):

            is_formatted = False
            if isinstance(argument_i, list) or isinstance(argument_i, tuple):
                try:
                    argument_i = np.asarray(argument_i)
                except ValueError:
                    index = tuple([slice(0, target_shape[a], 1) for a in range(0, n_axes, 1)])
                    tmp_argument_i = np.empty(shape=target_shape, dtype='O')
                    tmp_argument_i[index] = argument_i
                    argument_i = tmp_argument_i
                    del tmp_argument_i
                    is_formatted = True

            if not is_formatted:
                shape_argument_i = np.asarray(argument_i.shape, dtype=int)
                shape_in_argument_i = shape_argument_i[slice(0, n_axes, 1)]
                n_axis_in_argument_i = shape_in_argument_i.size
                if n_axis_in_argument_i != n_axes:
                    shape_new_argument_i = np.append(target_shape, shape_argument_i)
                    n_axis_in_new_argument_i = shape_new_argument_i.size
                    dtype = argument_i.dtype
                    argument_i_tmp = argument_i
                    for a in range(n_axes):
                        argument_i_tmp = np.expand_dims(argument_i_tmp, axis=0)
                    argument_i = np.empty(shape_new_argument_i, dtype=dtype)
                    index = np.empty(n_axis_in_new_argument_i, dtype=object)
                    index[:] = slice(None)
                    argument_i[tuple(index)] = argument_i_tmp

                elif (shape_in_argument_i == target_shape).all():
                    pass
                elif (shape_in_argument_i[shape_in_argument_i != target_shape] == 1).all():
                    shape_argument_i = shape_argument_i[slice(n_axes, None, 1)]
                    shape_new_argument_i = np.append(target_shape, shape_argument_i)
                    n_axis_in_new_argument_i = shape_new_argument_i.size
                    dtype = argument_i.dtype
                    argument_i_tmp = argument_i
                    argument_i = np.empty(shape_new_argument_i, dtype=dtype)
                    index = np.empty(n_axis_in_new_argument_i, dtype=object)
                    index[:] = slice(None)
                    argument_i[tuple(index)] = argument_i_tmp[tuple(index)]
                else:
                    shape_new_argument_i = np.append(target_shape, shape_argument_i)
                    n_axis_in_new_argument_i = shape_new_argument_i.size
                    dtype = argument_i.dtype
                    argument_i_tmp = argument_i
                    for a in range(n_axes):
                        argument_i_tmp = np.expand_dims(argument_i_tmp, axis=0)
                    argument_i = np.empty(shape_new_argument_i, dtype=dtype)
                    index = np.empty(n_axis_in_new_argument_i, dtype=object)
                    index[:] = slice(None)
                    argument_i[tuple(index)] = argument_i_tmp
        else:
            dtype_argument_i = type(argument_i)
            argument_i_tmp = argument_i
            argument_i = np.empty(target_shape, dtype=dtype_argument_i)
            argument_i[:] = argument_i_tmp

        dict_arguments[key_i] = argument_i

    return dict_arguments


def dict_to_key_array_and_value_array(dictionary):
    n_items = len(dictionary)
    keys = [None] * n_items
    values = np.empty(n_items, dtype=int)

    max_len_keys = 0
    i = 0
    for k, v in dictionary.items():
        keys[i] = k
        len_k = len(k)
        if len_k > max_len_keys:
            max_len_keys = len_k

        values[i] = v

        i += 1

    keys = np.asarray(keys, dtype=('<U' + str(max_len_keys)))
    return keys, values

