import numpy as np


def rotate_indexes(array, axes_rotating, m):

    # This function rotates the indexes of "array" on the "axes" by "m" indexes.
    # It returns None.

    shape_array = np.asarray(array.shape, dtype=int)
    n_axes = shape_array.size

    # format axes
    try:
        n_axes_rotating = len(axes_rotating)
        axes_rotating = np.asarray(axes_rotating, dtype=int)
        axes_rotating[axes_rotating < 0] += n_axes
    except TypeError:
        if axes_rotating < 0:
            axes_rotating += n_axes
        axes_rotating = np.asarray([axes_rotating], dtype=int)
        n_axes_rotating = 1

    n_indexes = shape_array[axes_rotating]
    # format m
    try:
        n_m = len(m)
        m = np.asarray(m, dtype=int)

        if n_m != n_axes_rotating:
            if n_m == 1:
                m_tmp = m[0]
                m = np.empty(n_axes_rotating, dtype=int)
                m[:] = m_tmp
                n_m = n_axes_rotating
            else:
                raise ValueError('Shapes of axes_inserting and variables_table_adding_axes must be equal')

    except TypeError:
        m = np.asarray([m], dtype=int)
        n_m = 1
        if n_m != n_axes_rotating:
            m_tmp = m[0]
            m = np.empty(n_axes_rotating, dtype=int)
            m[:] = m_tmp
            n_m = n_axes_rotating

    m %= n_indexes

    index_in = np.empty(n_axes, dtype=object)

    for a in axes_rotating:
        index_in[:] = slice(None)
        index_out = np.copy(index_in)

        index_in[a] = slice(n_indexes[a] - m[a], n_indexes[a], 1)
        array_out_0_to_m = np.copy(array[tuple(index_in)])

        index_in[a] = slice(0, n_indexes[a] - m[a], 1)
        index_out[a] = slice(m[a], n_indexes[a], 1)
        array[tuple(index_out)] = array[tuple(index_in)]

        index_out[a] = slice(0, m[a], 1)
        array[tuple(index_out)] = array_out_0_to_m

    return None
