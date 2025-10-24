
import math
import numpy as np
from . import combinations as cp_comps


def convert_to_float(num):

    if isinstance(num, (int, bool)):
        return float(num)
    elif isinstance(num, float):
        return num

    elif isinstance(num, (list, tuple)):

        out_num = [convert_to_float(num=num_i) for num_i in num]
        if isinstance(out_num, tuple):
            out_num = tuple(out_num)

        return out_num

    elif isinstance(num, np.ndarray):

        if num.dtype.kind in ('i', 'b', '?'):

            return num.astype(dtype='f')

        elif num.dtype.kind == 'f':

            return num

        elif num.dtype.kind == 'O':

            out_array = np.empty(num.shape, dtype='O')

            for comp_c in cp_comps.n_conditions_to_combinations_on_the_fly(n_conditions=num.shape, dtype='i'):

                indexes = tuple(comp_c.tolist())
                out_array[indexes] = convert_to_float(num=out_array[indexes])

            return out_array

        else:
            raise TypeError('num')
    else:
        raise TypeError('num')


def round_to_closest_int(num):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, np.int8, np.int16, np.int32, np.int64)):
        return num
    elif isinstance(num, (float, np.float16, np.float32, np.float64)):

        int_down = math.floor(num)
        int_up = math.ceil(num)

        if int_down == int_up:
            return int_down

        elif num < (int_down + 0.5):
            return int_down
        else:
            return int_up

    elif isinstance(num, bool):
        return int(num)

    elif isinstance(num, (list, tuple)):
        n = len(num)
        out_num = [round_to_closest_int(num[i]) for i in range(0, n, 1)]
        if isinstance(num, tuple):
            out_num = tuple(out_num)
        return out_num

    elif isinstance(num, np.ndarray):

        if num.dtype.kind == 'i':
            return num
        elif num.dtype.kind == 'f':

            int_down = np.floor(num).astype(dtype='i')
            int_up = np.ceil(num).astype(dtype='i')
            int_out = int_up
            indexes_out = num < (int_up - 0.5)
            int_out[indexes_out] = int_down[indexes_out]

            return int_out

        elif num.dtype.kind == '?':
            return num.astype(dtype='i')

        elif num.dtype.kind == 'O':

            out_array = np.empty(num.shape, dtype='O')

            for comp_c in cp_comps.n_conditions_to_combinations_on_the_fly(n_conditions=num.shape, dtype='i'):

                indexes = tuple(comp_c.tolist())
                out_array[indexes] = round_to_closest_int(num=out_array[indexes])

            return out_array

        else:
            raise TypeError('num')


    # elif isinstance(num, torch.Tensor):
    #
    #     if num.dtype.is_floating_point:
    #         int_down = num.floor().int()
    #         int_up = num.ceil().int()
    #         int_out = int_up
    #         indexes_out = num < (int_up - 0.5)
    #         int_out[indexes_out] = int_down[indexes_out]
    #
    #         return int_out
    #
    #     elif num.dtype.is_complex:
    #         raise TypeError('num')
    #     elif num.dtype == torch.bool:
    #         return num.int()
    #     else:
    #         return num

    else:
        raise TypeError('num')


def round_down_to_closest_int(num):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, np.int8, np.int16, np.int32, np.int64)):
        return num
    elif isinstance(num, (float, np.float16, np.float32, np.float64)):
        return math.floor(num)
    elif isinstance(num, bool):
        return int(num)
    elif isinstance(num, (list, tuple)):
        n = len(num)
        out_num = [round_down_to_closest_int(num[i]) for i in range(0, n, 1)]
        if isinstance(num, tuple):
            out_num = tuple(out_num)
        return out_num

    elif isinstance(num, np.ndarray):

        if num.dtype.kind == 'i':
            return num
        elif num.dtype.kind == 'f':
            return np.floor(num).astype(dtype='i')

        elif num.dtype.kind == '?':
            return num.astype(dtype='i')

        elif num.dtype.kind == 'O':

            out_array = np.empty(num.shape, dtype='O')
            for comp_c in cp_comps.n_conditions_to_combinations_on_the_fly(n_conditions=num.shape, dtype='i'):
                indexes = tuple(comp_c.tolist())
                out_array[indexes] = round_down_to_closest_int(num=out_array[indexes])

            return out_array

        else:

            raise TypeError('num')

    # elif isinstance(num, torch.Tensor):
    #
    #     if num.dtype.is_floating_point:
    #         return num.floor().int()
    #     elif num.dtype.is_complex:
    #         raise TypeError('num')
    #     elif num.dtype == torch.bool:
    #         return num.int()
    #     else:
    #         return num

    else:
        raise TypeError('num')


def round_up_to_closest_int(num):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, np.int8, np.int16, np.int32, np.int64)):
        return num
    elif isinstance(num, (float, np.float16, np.float32, np.float64)):
        return math.ceil(num)
    elif isinstance(num, bool):
        return int(num)
    elif isinstance(num, (list, tuple)):
        n = len(num)
        out_num = [round_up_to_closest_int(num[i]) for i in range(0, n, 1)]
        if isinstance(num, tuple):
            out_num = tuple(out_num)
        return out_num

    elif isinstance(num, np.ndarray):

        if num.dtype.kind == 'i':
            return num
        elif num.dtype.kind == 'f':
            return np.ceil(num).astype(dtype='i')

        elif num.dtype.kind == '?':
            return num.astype(dtype='i')

        elif num.dtype.kind == 'O':

            out_array = np.empty(num.shape, dtype='O')
            for comp_c in cp_comps.n_conditions_to_combinations_on_the_fly(n_conditions=num.shape, dtype='i'):
                indexes = tuple(comp_c.tolist())
                out_array[indexes] = round_up_to_closest_int(num=out_array[indexes])

            return out_array
        else:
            raise TypeError('num')

    # elif isinstance(num, torch.Tensor):
    #
    #     if num.dtype.is_floating_point:
    #         return num.ceil().int()
    #     elif num.dtype.is_complex:
    #         raise TypeError('num')
    #     elif num.dtype == torch.bool:
    #         return num.int()
    #     else:
    #         return num

    else:
        raise TypeError('num')


def round_to_closest_delta(num, delta=1):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :param delta:
    :type delta: int | float
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        return (round_to_closest_int(num=(np.asarray([num]) / delta)) * delta)[0].tolist()
    elif isinstance(num, np.ndarray):
        return round_to_closest_int(num=(num / delta)) * delta
    else:
        raise TypeError('num')


def round_down_to_closest_delta(num, delta=1):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :param delta:
    :type delta: int | float
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        return (round_down_to_closest_int(num=(np.asarray([num]) / delta)) * delta)[0].tolist()
    elif isinstance(num, np.ndarray):
        return round_down_to_closest_int(num=(num / delta)) * delta
    else:
        raise TypeError('num')


def round_up_to_closest_delta(num, delta=1):

    """

    :param num:
    :type num: int | float | bool | list | tuple | np.ndarray
    :param delta:
    :type delta: int | float
    :return: closest integers
    :rtype: int | list | tuple | np.ndarray
    """

    if isinstance(num, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        return (round_up_to_closest_int(num=(np.asarray([num]) / delta)) * delta)[0].tolist()
    elif isinstance(num, np.ndarray):
        return round_up_to_closest_int(num=(num / delta)) * delta
    else:
        raise TypeError('num')


def convert_to_int_or_float(num):

    if isinstance(num, int):
        return num
    else:
        str_num = str(num)
        list_num = str_num.split('.')
        len_num = len(list_num)
        if len_num == 1:
            return int(num)
        elif len_num == 2:
            list_decimal = list_num[1].split('e')
            len_decimal = len(list_decimal)
            if len_decimal == 1:
                if int(list_num[1]) == 0:
                    return int(num)
                elif isinstance(num, float):
                    return num
                else:
                    return float(num)
            elif int(list_decimal[1]) < 0:
                if isinstance(num, float):
                    return num
                else:
                    return float(num)
            elif int(list_decimal[1]) > 0:
                return round_to_closest_int(num)

        else:
            raise TypeError('num')


def if_equal_to_the_nearest_int_convert_to_int_else_float(num):
    if isinstance(num, int):
        return num
    else:
        num_rounded = round_to_closest_int(num)
        if num_rounded == num:
            return num_rounded
        else:
            return float(num)


def factors_of_x(x, y=1):

    x = convert_to_int_or_float(x)
    y = convert_to_int_or_float(y)

    if isinstance(x, float):
        raise TypeError('Type of x must be int')

    if isinstance(y, float):
        raise TypeError('Type of y must be int')

    factors = np.empty(0, dtype='i')

    for i in range(y, x + 1):
        if x % i == 0:
            factors = np.append(factors, i)

    return factors


def prod(numbers, start=1):

    product = start
    if isinstance(numbers, np.ndarray):
        for num_i in numbers.tolist():
            product *= num_i
    else:
        for num_i in numbers:
            product *= num_i

    return product


def gamma(z):
    
    print('scipy.special.gamma(z) is more efficient')
    
    pos_inf = 100
    n_dx = 10000000
    dx = pos_inf / n_dx

    if not isinstance(z, np.ndarray):
        z = np.asarray(z)

    n_axes_z = len(z.shape)
    n_axes_x = n_axes_z + 1
    axis_delta = n_axes_x - 1
    x = np.arange(0, pos_inf, dx)
    while len(x.shape) < n_axes_x:
        x = np.expand_dims(x, axis=0)

    if n_axes_z > 0:
        z = np.expand_dims(z, axis_delta)

    with np.errstate(divide='ignore'):
        return np.sum(
            np.power(x, z - 1) * np.power(math.e, -x) * dx, axis=axis_delta)


def is_nan(x):
    return x != x


def is_not_nan(x):
    return x == x
