from . import plt
from .parameters import my_dpi
from .. import directory as cp_directory
import os


__all__ = ['save_figures']


def save_figures(id_figures=None, directories=None, formats=None, close_figures=False):

    if id_figures is None:
        id_figures = plt.get_fignums()

    n_figures = len(id_figures)

    if directories is None:
        directories = [str(d) for d in id_figures]
    else:
        if not (isinstance(directories, list) or
                isinstance(directories, tuple)):
            directories = [directories]

        n_directories = len(directories)
        if n_figures != n_directories:
            raise ValueError('n_figures and n_directories must be equal.\n'
                             'Now, n_figures = {} and n_directories = {}'.format(n_figures, n_directories))

    if formats is None:
        formats = ['svg']

    n_formats = len(formats)
    for i in range(n_figures):

        figure_i = plt.figure(id_figures[i])

        root_i = os.path.dirname(directories[i])

        if len(root_i) > 0:
            os.makedirs(root_i, exist_ok=True)

        for j in range(n_formats):

            directory_i_j = cp_directory.add_extension(directories[i], formats[j])
            plt.savefig(directory_i_j, format=formats[j], dpi=my_dpi)

        if close_figures:
            figure_i.clf()
            plt.close(fig=figure_i)
            figure_i = None
