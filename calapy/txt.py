

import numpy as np
import csv
import pandas as pd
import os
import glob
from itertools import (takewhile, repeat)
from . import array as cp_array
from . import directory as cp_directory
from . import format as cp_format
from . import combinations as cp_combinations


def text_to_txt_file(text, filename, encoding=None):
    """Write a new text file with some text.

    :param text: The text to write in the new text file.
    :type text: str
    :param filename: The path of the new text file.
    :type filename: str
    :param encoding: Argument of the Python built-in function open().
    :type encoding: str | None
    :return: None.
    :rtype: None
    """

    with open(filename, mode='w', newline=None, encoding=encoding) as file:
        file.write(text)


def txt_file_to_lines(file_path, remove_empty_lines=True, encoding=None):

    """Get the lines of a text file as a list of strings.

    :param file_path: The path of the text file.
    :type file_path: str
    :param remove_empty_lines: If True (default), it removes all empty lines. If False, it keeps all lines.
    :type remove_empty_lines: bool
    :param encoding: Argument of the Python built-in function open().
    :type encoding: str | None
    :return: The lines of the text file as list of strings.
    :rtype: list[str]
    """

    with open(file_path, mode='r', newline=None, encoding=encoding) as reader:
        lines = reader.readlines()

    L = len(lines)
    for l in range(L - 1, -1, -1):
        if lines[l] == '\n':
            if remove_empty_lines:
                # remove extra empty lines
                lines.pop(l)
            else:
                # remove newline characters "\n" from all lines
                lines[l] = ''
        else:
            # remove newline characters "\n" from all lines
            lines[l] = lines[l].removesuffix('\n')

    return lines


def lines_to_txt_file(lines, filename, encoding=None):

    L = len(lines)

    with open(filename, mode='w', newline=None, encoding=encoding) as file:
        # file.writelines(lines)
        for l in range(0, L, 1):
            file.write(lines[l] + '\n')


def lines_to_txt_files(lines, filenames, encoding=None):

    n_files = len(lines)
    for f in range(n_files):
        lines_to_txt_file(lines=lines[f], filename=filenames[f], encoding=encoding)


def lines_to_csv_file(lines, directory, headers=None, delimiter=None, encoding=None):
    if delimiter is None:
        delimiter = ','

    with open(directory, 'w', newline='', encoding=encoding) as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        if headers is not None:
            writer.writerow(headers)
        writer.writerows(lines)


def array_to_csv_files(
        array_in, axis_rows, axis_columns, conditions_of_directories, headers=None, delimiter=None, encoding=None):

    if axis_rows > axis_columns:
        array = np.swapaxes(array_in, axis_rows, axis_columns)
        axis_rows, axis_columns = axis_columns, axis_rows
    elif axis_rows < axis_columns:
        array = array_in
    else:
        raise ValueError('axis_rows, axis_columns')

    shape_array = np.asarray(array.shape, dtype='i')
    n_axes = shape_array.size
    indexes_array = np.empty(n_axes, dtype='O')
    indexes_array[axis_rows] = slice(0, shape_array[axis_rows], 1)
    indexes_array[axis_columns] = slice(0, shape_array[axis_columns], 1)

    axes_array = np.arange(0, n_axes, 1)
    axes_table = np.asarray([axis_rows, axis_columns], dtype='i')
    axes_files = np.where(np.logical_not(cp_array.samples_in_arr1_are_in_arr2(axes_array, axes_table, axis=0)))[0]

    if delimiter is None:
        delimiter = ','

    # for combination_indexes_i, combination_directories_i in cp_combinations.conditions_to_combinations_on_the_fly(
    #         conditions_of_directories, dtype='U', order_outputs='iv'):
    for combination_indexes_i, directory_i in cp_directory.conditions_to_directories_on_the_fly(
            conditions_of_directories, order_outputs='iv'):

        indexes_array[axes_files] = combination_indexes_i
        array_i = array[tuple(indexes_array)]

        dirname_i = os.path.dirname(directory_i)
        os.makedirs(dirname_i, exist_ok=True)
        with open(directory_i, 'w', newline='', encoding=encoding) as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            if headers is not None:
                writer.writerow(headers)
            writer.writerows(array_i)


def csv_file_to_lines(directory, delimiter=None, encoding=None):

    if delimiter is None:
        delimiter = ','

    with open(directory, newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        lines = list(reader)
    return lines


def count_rows(filename, encoding=None):
    f = open(filename, 'rb', encoding=encoding)
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen)


def count_columns(filename, rows, delimiter=None, encoding=None):

    if delimiter is None:
        delimiter = ','

    try:
        0 in rows
    except TypeError:
        rows = [rows]

    max_rows = max(rows)
    n_columns = [0 in range(0, len(rows), 1)]

    with open(filename, newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row_r in enumerate(reader):

            if row_r > max_rows:
                break

            if row_r in rows:
                n_columns = len(row_r)

    return max(n_columns)


def csv_file_to_array(
        filename, rows=None, columns=None, delimiter=None, dtype=None, encoding=None):

    if delimiter is None:
        delimiter = ','

    if isinstance(rows, slice):
        start = rows.start
        if start is None:
            start = 0

        stop = rows.stop
        if stop is None:
            stop = count_rows(filename=filename, encoding=encoding)

        step = rows.step
        if step is None:
            step = 1

        keeping_rows = range(start, stop, step)
    else:
        keeping_rows = rows

    if isinstance(columns, slice):
        start = columns.start
        if start is None:
            start = 0

        stop = columns.stop
        if stop is None:
            stop = count_columns(filename=filename, rows=keeping_rows, encoding=encoding)

        step = columns.step
        if step is None:
            step = 1

        keeping_columns = range(start, stop, step)
    else:
        keeping_columns = columns

    exclude_rows = None if keeping_rows is None else lambda x: x not in keeping_rows
    table = pd.read_csv(
        filename,
        skiprows=exclude_rows, usecols=keeping_columns,
        sep=delimiter, dtype=dtype, encoding=encoding, index_col=False, header=None)

    return table.to_numpy()
    # with open(filename, newline='', encoding=encoding) as csvfile:
    #     reader = csv.reader(csvfile, delimiter=delimiter)
    #     # for line in reader:
    #     #     print()
    #     # lines = list(reader)
    #     table = np.asarray(list(reader), dtype='O')
    #
    # if rows is None:
    #     if columns is None:
    #         return table.astype(dtype)
    #     else:
    #         rows = slice(0, table.shape[0], 1)
    #         indexes = tuple([rows, columns])
    #         return table[indexes].astype(dtype)
    # else:
    #     if columns is None:
    #         columns = slice(0, table.shape[1], 1)
    #
    #     indexes = tuple([rows, columns])
    #     return table[indexes].astype(dtype)


def csv_file_to_arrays(filename, rows, columns, delimiter=None, dtype=None, encoding=None):
    try:
        n_arrays_from_rows = len(rows)
    except TypeError:
        rows = [rows]
        n_arrays_from_rows = 1
    try:
        n_arrays_from_columns = len(columns)
    except TypeError:
        columns = [columns]
        n_arrays_from_columns = 1

    if n_arrays_from_rows != n_arrays_from_columns:
        if n_arrays_from_rows == 1:
            rows = list(rows) * n_arrays_from_columns
            n_arrays_from_rows = n_arrays_from_columns
        elif n_arrays_from_columns == 1:
            columns = list(columns) * n_arrays_from_rows
            n_arrays_from_columns = n_arrays_from_rows
        else:
            raise ValueError(
                'The following assumption is not met:\n'
                '\t n_arrays_from_rows' + ' \u003D ' + 'n_arrays_from_columns')

    n_arrays = n_arrays_from_rows
    try:
        n_dtypes = len(dtype)
    except TypeError:
        dtype = [dtype]
        n_dtypes = 1
    if n_arrays != n_dtypes:
        if n_dtypes == 1:
            dtype = list(dtype) * n_arrays
        else:
            raise ValueError(
                'The following assumption is not met:\n'
                '\t(n_dtypes \u003D n_arrays_from_rows) OR (n_dtypes \u003D n_arrays_from_columns)')

    if delimiter is None:
        delimiter = ','

    with open(filename, newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        table = np.asarray(list(reader), dtype='U')

    n_axes_per_text = 2
    arrays = [None] * n_arrays
    for a in range(n_arrays):
        indexes = tuple(cp_format.numeric_indexes_to_slice([rows[a], columns[a]]))
        if (dtype[a] is None) or (dtype[a] == str) or (dtype[a] == 'U'):
            arrays[a] = table[indexes]
        else:
            arrays[a] = table[indexes].astype(dtype[a])
    return arrays


def conditions_of_csv_files_to_arrays(
        conditions_of_directories, rows, columns, delimiter=None, dtype=None, encoding=None):

    n_axes_directories = len(conditions_of_directories)
    n_conditions_directories = np.empty(n_axes_directories, dtype='i')
    for i in range(n_axes_directories):
        n_conditions_directories[i] = len(conditions_of_directories[i])
    # logical_indexes_conditions = n_conditions_directories > 1
    combinations_directories = cp_combinations.n_conditions_to_combinations(n_conditions_directories)
    n_combinations_directories = combinations_directories.shape[0]
    axes_directories_squeezed = n_conditions_directories > 1
    n_axes_directories_squeezed = np.sum(axes_directories_squeezed)

    try:
        n_arrays_from_rows = len(rows)
    except TypeError:
        rows = [rows]
        n_arrays_from_rows = 1
    try:
        n_arrays_from_columns = len(columns)
    except TypeError:
        columns = [columns]
        n_arrays_from_columns = 1

    if n_arrays_from_rows != n_arrays_from_columns:
        if n_arrays_from_rows == 1:
            rows = list(rows) * n_arrays_from_columns
            n_arrays_from_rows = n_arrays_from_columns
        elif n_arrays_from_columns == 1:
            columns = list(columns) * n_arrays_from_rows
            n_arrays_from_columns = n_arrays_from_rows
        else:
            raise ValueError(
                'The following assumption is not met:\n'
                '\t n_arrays_from_rows' + ' \u003D ' + 'n_arrays_from_columns')

    n_arrays = n_arrays_from_rows
    try:
        n_dtypes = len(dtype)
    except TypeError:
        dtype = [dtype]
        n_dtypes = 1
    if n_arrays != n_dtypes:
        if n_dtypes == 1:
            dtype = list(dtype) * n_arrays
        else:
            raise ValueError(
                'The following assumption is not met:\n'
                '\t(n_dtypes \u003D n_arrays_from_rows) OR (n_dtypes \u003D n_arrays_from_columns)')

    n_axes_per_csv = 2
    indexes = [None] * n_arrays  # type: list
    arrays = [None] * n_arrays  # type: list
    for a in range(n_arrays):
        indexes[a] = tuple(cp_format.numeric_indexes_to_slice([rows[a], columns[a]]))

    directory_csv_d = os.path.join(*[
        conditions_of_directories[i][0] for i in range(n_axes_directories)])

    if delimiter is None:
        delimiter = ','

    with open(directory_csv_d, newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        # array_csv_d = list(reader)
        array_csv_d = np.asarray(list(reader), dtype='U')

    indexes_out = [None for a in range(n_arrays)]  # type: list
    convert_to_U = [None for a in range(n_arrays)]  # type: list
    for a in range(n_arrays):
        if (dtype[a] is None) or (dtype[a] == 'U') or (dtype[a] == str):
            dtype[a] = 'O'
            convert_to_U[a] = True

        array_a_d = array_csv_d[indexes[a]]
        shape_array_a_d = np.asarray(array_a_d.shape, dtype=int)
        shape_array_a = np.append(
            n_conditions_directories[axes_directories_squeezed], shape_array_a_d)

        arrays[a] = np.empty(shape_array_a, dtype=dtype[a])
        n_axes_a = shape_array_a.size
        indexes_out[a] = np.empty(n_axes_a, dtype='O')
        indexes_out[a][:] = slice(None)

    for d in range(n_combinations_directories):

        directory_csv_d = os.path.join(*[
            conditions_of_directories[i][combinations_directories[d, i]] for i in range(n_axes_directories)])

        with open(directory_csv_d, newline='', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            array_csv_d = np.asarray(list(reader), dtype='U')

        for a in range(n_arrays):

            indexes_out[a][slice(0, n_axes_directories_squeezed, 1)] = (
                combinations_directories[d, axes_directories_squeezed])

            arrays[a][tuple(indexes_out[a])] = array_csv_d[indexes[a]]  # .astype(dtype[a])

    for a in range(n_arrays):
        if convert_to_U[a]:
            arrays[a] = arrays[a].astype('U')

    return arrays


def n_csv_files_to_array_old(
        directories_csv_files, names_csv_files_in_directories=False,
        rows=slice(None), columns=slice(None), data_type=None, delimiter=None, encoding=None):

    print('using the funtion n_csv_files_to_array().\n'
          'In the future versions of ccalafiore, it will be removed.\n'
          'Consider using conditions_of_csv_files_to_arrays()')
    array = None

    n_axes_directories = len(directories_csv_files)
    n_conditions_directories = np.empty(n_axes_directories, dtype=int)
    for a in range(n_axes_directories):
        n_conditions_directories[a] = len(directories_csv_files[a])

    logical_indexes_conditions = n_conditions_directories > 1

    combinations_directories = cp_combinations.n_conditions_to_combinations(n_conditions_directories)
    n_combinations_directories = combinations_directories.shape[0]

    if delimiter is None:
        delimiter = ','

    for d in range(n_combinations_directories):

        directory = directories_csv_files[0][combinations_directories[d, 0]]
        for a in range(1, n_axes_directories):
            directory = os.path.join(
                directory, directories_csv_files[a][combinations_directories[d, a]])

        if names_csv_files_in_directories:
            files_per_directory = [directory]
        else:
            files_per_directory = glob.glob(os.path.join(directory, '*.csv'))

        n_files_per_directory = len(files_per_directory)

        start_row_f_file = 0
        for f in range(n_files_per_directory):

            array_per_file = np.genfromtxt(files_per_directory[f], delimiter=delimiter, dtype=data_type)[rows, columns]

            if array is None:

                shape_array_per_file = np.asarray(array_per_file.shape, dtype=int)

                shape_array = np.asarray([
                    *n_conditions_directories[logical_indexes_conditions],
                    n_files_per_directory * shape_array_per_file[0],
                    *shape_array_per_file[1:]
                ], dtype=int)

                array = np.empty(shape_array, dtype=data_type)

            end_row_f_file = (f + 1) * shape_array_per_file[0]

            indexes_array = tuple(
                [*combinations_directories[d][logical_indexes_conditions],
                 slice(start_row_f_file, end_row_f_file)])

            array[indexes_array] = array_per_file
            start_row_f_file = end_row_f_file

    return array
