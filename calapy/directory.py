import os
import sys
from . import combinations as cp_combinations
from . import maths as cp_maths


def n_directories_up(directory, n=1):
    
    while n > 0:
        directory = os.path.dirname(directory)
        n -= 1

    return directory


def get_root_directory_of_running_script():
    directory_scripts = os.path.realpath(sys.argv[0])
    root_directory = os.path.dirname(directory_scripts)
    return root_directory


def conditions_to_directories_on_the_fly(conditions_directories, order_outputs='v'):
    """
    Parameters
    ----------
    order_outputs : str or sequence of str, optional
        The desired outputs. Accepted values are "v", "i" or any combination of them like "vi", "iv" (default is "vi").
        "v" stands for combination_values_i and "i" for combination_indexes_i.
    """
    order_accepted_values = 'vi'
    if order_outputs is None:
        order_outputs = 'v'
        n_outputs = 2
    else:
        n_outputs = len(order_outputs)
        if n_outputs < 1:
            raise ValueError('order_outputs')
        else:
            for o in range(n_outputs):
                if not (order_outputs[o] in order_accepted_values):
                    raise ValueError('order_outputs')
    if n_outputs > 1:
        outputs_i = [None] * n_outputs  # type: list

    for combinations_i in cp_combinations.conditions_to_combinations_on_the_fly(
            conditions_directories, dtype='U', order_outputs=order_outputs):
        if n_outputs > 1:
            for o in range(n_outputs):
                if order_outputs[o] == 'v':
                    combination_directories_i = combinations_i[o]
                    directory_i = os.path.join(*combination_directories_i)
                    outputs_i[o] = directory_i
                elif order_outputs[o] == 'i':
                    combination_indexes_i = combinations_i[o]
                    outputs_i[o] = combination_indexes_i
            yield outputs_i
        elif n_outputs == 1:
            if order_outputs[0] == 'v':
                combination_directories_i = combinations_i
                directory_i = os.path.join(*combination_directories_i)
                yield directory_i
            elif order_outputs[0] == 'i':
                combination_indexes_i = combinations_i
                yield combination_indexes_i


def conditions_to_directories(conditions_directories, order_outputs='v'):

    n_conditions = cp_combinations.conditions_to_n_conditions(conditions_directories)
    n_directories = cp_maths.prod(n_conditions)
    # n_variables = len(n_conditions)

    order_accepted_values = 'vi'
    if order_outputs is None:
        order_outputs = 'v'
        n_outputs = 2
    else:
        n_outputs = len(order_outputs)
        if n_outputs < 1:
            raise ValueError('order_outputs')
        else:
            for o in range(n_outputs):
                if not (order_outputs[o] in order_accepted_values):
                    raise ValueError('order_outputs')
    if n_outputs > 1:
        outputs_i = [[None] * n_directories] * n_outputs  # type: list
    else:
        outputs_i = [None] * n_directories  # type: list

    # directories = [None] * n_directories  # type: list
    i = -1
    for combinations_i in conditions_to_directories_on_the_fly(conditions_directories, order_outputs=order_outputs):
        i += 1
        # directories[i] = directory_i

        if n_outputs > 1:
            for o in range(n_outputs):
                if order_outputs[o] == 'v':
                    combination_directories_i = combinations_i[o]
                    outputs_i[o][i] = combination_directories_i
                elif order_outputs[o] == 'i':
                    combination_indexes_i = combinations_i[o]
                    outputs_i[o][i] = combination_indexes_i
        elif n_outputs == 1:
            if order_outputs[0] == 'v':
                combination_directories_i = combinations_i
                outputs_i[i] = combination_directories_i
            elif order_outputs[0] == 'i':
                combination_indexes_i = combinations_i
                outputs_i[i] = combination_indexes_i

    return outputs_i


def get_extension(directory, point=False):
    basename_with_extension = os.path.basename(directory)
    basename_no_extension, extension_with_point = os.path.splitext(basename_with_extension)
    if point:
        return extension_with_point
    else:
        extension_no_point = extension_with_point[slice(1, None, 1)]
        return extension_no_point


def remove_extension(directory):
    dirname = os.path.dirname(directory)
    basename_with_extension = os.path.basename(directory)
    basename_no_extension = os.path.splitext(basename_with_extension)[0]
    directory_no_extension = os.path.join(dirname, basename_no_extension)
    return directory_no_extension


def add_extension(directory: str, extension: str):

    if extension[0] != '.':
        extension_formatted = '.' + extension
    else:
        extension_formatted = extension

    len_directory = len(directory)
    len_extension_formatted = len(extension_formatted)
    idx = len_directory - len_extension_formatted
    if idx > 0:
        if directory[slice(idx, len_directory, 1)] == extension_formatted:
            directory_with_extension = directory
        else:
            directory_with_extension = directory + extension_formatted
    else:
        directory_with_extension = directory + extension_formatted

    return directory_with_extension


def replace_extension(directory, extension):
    directory_no_extension = remove_extension(directory)
    directory_with_extension = add_extension(directory_no_extension, extension)
    return directory_with_extension


def replace_extensions(directories, extension):
    n_directories = len(directories)
    for d in range(n_directories):
        directories[d] = replace_extension(directories[d], extension)
    return directories
