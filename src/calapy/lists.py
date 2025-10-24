import numpy as np


def pad_list(x, size, align, pad_type='constant', pad_value=None, dtype=None):

    # pad_types in https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_BorderTypes.html
    # pad_type can only be 'constant' or 'replicate'

    n = len(x)

    lower_align = align.lower()

    if n < size:

        lower_pad_type = pad_type.lower()
        if not pad_type.lower() in ['constant', 'replicate']:
            raise ValueError('pad_type')

        if dtype is None:
            dtype = np.asarray(x[slice(0, 2, 1)]).dtype

        out = np.empty(size, dtype=dtype)

        if lower_align in ['left', 'first']:

            if lower_pad_type == 'replicate':
                pad_value = x[-1]

            out[slice(0, n, 1)] = x

            out[slice(n, size, 1)] = pad_value

        elif lower_align in ['right', 'last']:

            if lower_pad_type == 'replicate':
                pad_value = x[0]

            tmp = size - n

            out[slice(0, tmp, 1)] = pad_value

            out[slice(tmp, size, 1)] = x

        else:
            raise ValueError('align')

    elif n > size:
        if lower_align in ['left', 'first']:

            # out[slice(0, size, 1)] = x[slice(0, size, 1)]
            out = np.asarray(x[slice(0, size, 1)], dtype=dtype)

        elif lower_align in ['right', 'last']:

            # out[slice(0, size, 1)] = x[slice(n - size, n, 1)]
            out = np.asarray(x[slice(n - size, n, 1)], dtype=dtype)
        else:
            raise ValueError('align')

    elif n == size:
        if lower_align in ['left', 'first', 'right', 'last']:
            out = np.asarray(x, dtype=dtype)
        else:
            raise ValueError('align')
    else:
        out = None

    return out
