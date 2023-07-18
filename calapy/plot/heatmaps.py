import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
from . import plt
from .parameters import my_dpi
from . import cmaps as cp_cmaps
from .. import combinations as cp_combinations
from .. import format as cp_format
from .. import check as cp_check
from .. import maths as cp_maths


# todo: 1) add option save_figs to save the figures as they are made

# todo: 2) add option close_figs to close and delete the figures as they are saved

# todo: 3) add multi directories

# todo: 4) if axes are shared, then chose whether to draw tick labels:
#  this can be done here ".tick_params(axis='both', labelbottom=True, labelleft=True)"

# todo: 5) fix bug of min_x, min_y, max_x, max_y


def single_figure_single_plot(
        data, dict_axes,
        annot=False, array_annot=None, fmt_annot='{:.2f}', colors_annot=None,
        font_size_annot=None, threshold_change_color_annot=None,
        cmap=None, n_levels_cmap=None,
        add_cbar=True, label_cbar=None, font_size_label_cbar=None,
        labels_ticks_cbar=None, ticks_cbar=None, n_ticks_cbar=None,
        font_size_labels_ticks_cbar=None, rotation_labels_ticks_cbar=None,
        title=None, font_size_title=None, rotation_title=None,
        label_x=None, font_size_label_x=None, rotation_label_x=None,
        label_y=None, font_size_label_y=None, rotation_label_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_size_labels_ticks_x=None, rotation_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_size_labels_ticks_y=None, rotation_labels_ticks_y=None,
        min_cmap=None, max_cmap=None,
        ax=None, ratio_plot=None, maximum_n_pixels_per_plot=500, tight_layout=True):

    shape_data = np.asarray(data.shape, dtype='i')
    n_axes = shape_data.size
    if n_axes != 2:
        raise ValueError('data must have either 2 axes')

    keys_axes_expected = np.asarray(['y', 'x'], dtype='U')
    values_axes_expected = np.arange(n_axes)

    keys_axes, axes_data = cp_format.dict_to_key_array_and_value_array(dict_axes)
    axes_negative = axes_data < 0
    axes_data[axes_negative] += n_axes
    for k in keys_axes[axes_negative]:
        dict_axes[k] += n_axes

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        dict_axes, keys_axes_expected, values_axes_expected, name_dictionary='dict_axes')
    cp_check.values_are_not_repeated(dict_axes, name_dictionary='dict_axes')

    if dict_axes['x'] < dict_axes['y']:
        data = data.T
        dict_axes = dict(zip(keys_axes_expected, values_axes_expected))
        shape_data = np.asarray(data.shape, dtype='i')

    if ax is None:
        figures_existing = plt.get_fignums()
        n_figures_new = 1
        i = 0
        f = 0
        id_figure = None
        while f < n_figures_new:
            if i in figures_existing:
                pass
            else:
                id_figure = i
                f += 1
            i += 1

        if ratio_plot is None:
            ratio_plot = {}
            if data.shape[dict_axes['x']] > data.shape[dict_axes['y']]:
                ratio_plot['x'] = 1
                ratio_plot['y'] = data.shape[dict_axes['y']] / data.shape[dict_axes['x']]
            elif data.shape[dict_axes['x']] < data.shape[dict_axes['y']]:
                ratio_plot['x'] = data.shape[dict_axes['x']] / data.shape[dict_axes['y']]
                ratio_plot['y'] = 1
            else:
                ratio_plot['x'] = 1
                ratio_plot['y'] = 1

        fig = plt.figure(
            num=id_figure, frameon=False, dpi=my_dpi,
            figsize=((maximum_n_pixels_per_plot * ratio_plot['x']) / my_dpi, (maximum_n_pixels_per_plot * ratio_plot['y']) / my_dpi))
        ax = plt.gca()
    else:
        fig = ax.figure

    # Plot the heatmap
    # im = ax.imshow(data)
    if isinstance(cmap, cp_cmaps.ColorMap):
        cmap = matplotlib.colors.ListedColormap(cmap.array, N=n_levels_cmap)
    elif isinstance(cmap, (list, tuple, np.ndarray)):
        cmap = matplotlib.colors.ListedColormap(cmap, N=n_levels_cmap)
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        pass
    elif (cmap is not None) and (n_levels_cmap is not None):
        cmap = plt.get_cmap(cmap, n_levels_cmap)

    im = ax.imshow(data, cmap=cmap, vmin=min_cmap, vmax=max_cmap)

    # Create colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    if add_cbar:
        # cbar = ax.figure.colorbar(
        #     im, ax=ax, ticks=np.arange(levels_cmap))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)

        if labels_ticks_cbar is None:
            if ticks_cbar is None:
                if n_ticks_cbar is None:
                    pass
                else:
                    max_cbar = data.max()
                    min_cbar = data.min()
                    n_labels_ticks_cbar = n_ticks_cbar
                    delta_cbar = (max_cbar - min_cbar) / (n_labels_ticks_cbar - 1)
                    tick_cbar_lower = min_cbar
                    tick_cbar_higher = max_cbar
                    ticks_cbar = np.arange(
                        tick_cbar_lower, tick_cbar_higher + delta_cbar, delta_cbar)
        else:
            if ticks_cbar is None:
                max_cbar = data.max()
                min_cbar = data.min()
                n_labels_ticks_cbar = len(labels_ticks_cbar)
                delta_cbar = (max_cbar - min_cbar) / n_labels_ticks_cbar
                tick_cbar_lower = min_cbar + (delta_cbar / 2)
                tick_cbar_higher = max_cbar - (delta_cbar / 2)
                ticks_cbar = np.arange(tick_cbar_lower, tick_cbar_higher + delta_cbar, delta_cbar)

        cbar = ax.figure.colorbar(im, cax=cax, orientation='vertical', ticks=ticks_cbar)

        vertical_alignment_labels_ticks_cbar = 'center'
        horizontal_alignment_labels_ticks_cbar = 'left'
        # if rotation_labels_ticks_cbar is None:
        #     horizontal_alignment_labels_ticks_cbar = 'left'
        # elif 45 <= (abs(rotation_labels_ticks_cbar) % 180) <= 135:
        #     horizontal_alignment_labels_ticks_cbar = 'left'
        # else:
        #     horizontal_alignment_labels_ticks_cbar = 'left'

        if labels_ticks_cbar is None:
            if (font_size_labels_ticks_cbar is not None) or (rotation_labels_ticks_cbar is not None):
                cbar.ax.tick_params(
                    labelsize=font_size_labels_ticks_cbar, labelrotation=rotation_labels_ticks_cbar)
                    # horizontalalignment=horizontal_alignment_labels_ticks_cbar,
                    # verticalalignment=vertical_alignment_labels_ticks_cbar)
        else:
            cbar.ax.set_yticklabels(
                labels=labels_ticks_cbar, fontsize=font_size_labels_ticks_cbar, rotation=rotation_labels_ticks_cbar,
                horizontalalignment=horizontal_alignment_labels_ticks_cbar,
                verticalalignment=vertical_alignment_labels_ticks_cbar)

        if label_cbar is not None:
            cbar.ax.set_ylabel(
                label_cbar, fontsize=font_size_label_cbar, rotation=-90,
                horizontalalignment='left', verticalalignment='center')

    if title is not None:
        ax.set_title(title, fontsize=font_size_title, rotation=rotation_title)
    if label_x is not None:
        ax.set_xlabel(label_x, fontsize=font_size_label_x, rotation=rotation_label_x)
    if label_y is not None:
        ax.set_ylabel(label_y, fontsize=font_size_label_y, rotation=rotation_label_y)

    ticks_x_are_applied = False
    if labels_ticks_x is None:
        # max_x = data.shape[dict_axes['x']] - 1
        if ticks_x is None:
            if n_ticks_x is None:
                pass
            else:
                max_x = data.shape[dict_axes['x']] - 1
                min_x = 0
                n_labels_ticks_x = n_ticks_x
                delta_x = (max_x - min_x) / (n_labels_ticks_x - 1)
                tick_x_lower = min_x
                tick_x_higher = max_x
                ticks_x = np.arange(tick_x_lower, tick_x_higher + (delta_x / 2), delta_x)
                ax.set_xticks(ticks_x)
                ticks_x_are_applied = True
        else:
            ax.set_xticks(ticks_x)
            ticks_x_are_applied = True

        if stagger_labels_ticks_x or (font_size_labels_ticks_x is not None) or (rotation_labels_ticks_x is not None):
            fig.canvas.draw()
            tmp_labels_ticks_x = ax.get_xticklabels()[1:-1:1]
            n_labels_ticks_x = len(tmp_labels_ticks_x)
            labels_ticks_x = [None for i in range(0, n_labels_ticks_x, 1)]

            if ticks_x is None:
                ticks_x = [None for i in range(0, n_labels_ticks_x, 1)]
                get_ticks_x = True
            else:
                get_ticks_x = False

            for l, label_l in enumerate(tmp_labels_ticks_x):
                labels_ticks_x[l] = label_l.get_text()

                if get_ticks_x:
                    ticks_x[l] = label_l.get_position()[0]

    if labels_ticks_x is not None:
        if not ticks_x_are_applied:
            if ticks_x is None:
                max_x = data.shape[dict_axes['x']] - 1
                min_x = 0
                n_labels_ticks_x = len(labels_ticks_x)
                delta_x = (max_x - min_x) / (n_labels_ticks_x - 1)
                tick_x_lower = min_x
                tick_x_higher = max_x
                ticks_x = np.arange(tick_x_lower, tick_x_higher + (delta_x / 2), delta_x)

            ax.set_xticks(ticks_x)
        if stagger_labels_ticks_x:
            labels_ticks_x = (
                [str(l) if not i % 2 else '\n' + str(l) for i, l in enumerate(labels_ticks_x)])
        ax.set_xticklabels(
            labels_ticks_x, fontsize=font_size_labels_ticks_x,
            rotation=rotation_labels_ticks_x)

    ticks_y_are_applied = False
    if labels_ticks_y is None:
        # max_y = data.shape[dict_axes['y']] - 1
        if ticks_y is None:
            if n_ticks_y is None:
                pass
            else:
                max_y = data.shape[dict_axes['y']] - 1
                min_y = 0
                n_labels_ticks_y = n_ticks_y
                delta_y = (max_y - min_y) / (n_labels_ticks_y - 1)
                tick_y_lower = min_y
                tick_y_higher = max_y
                ticks_y = np.arange(tick_y_lower, tick_y_higher + (delta_y / 2), delta_y)
                ax.set_yticks(ticks_y)
                ticks_y_are_applied = True

        else:
            ax.set_yticks(ticks_y)
            ticks_y_are_applied = True

        if stagger_labels_ticks_y or (font_size_labels_ticks_y is not None) or (rotation_labels_ticks_y is not None):
            fig.canvas.draw()
            tmp_labels_ticks_y = ax.get_yticklabels()[1:-1:1]
            n_labels_ticks_y = len(tmp_labels_ticks_y)
            labels_ticks_y = [None for i in range(0, n_labels_ticks_y, 1)]

            if ticks_y is None:
                ticks_y = [None for i in range(0, n_labels_ticks_y, 1)]
                get_ticks_y = True
            else:
                get_ticks_y = False

            for l, label_l in enumerate(tmp_labels_ticks_y):
                labels_ticks_y[l] = label_l.get_text()

                if get_ticks_y:
                    ticks_y[l] = label_l.get_position()[1]

    if labels_ticks_y is not None:
        if not ticks_y_are_applied:
            if ticks_y is None:
                max_y = data.shape[dict_axes['y']] - 1
                min_y = 0
                n_labels_ticks_y = len(labels_ticks_y)
                delta_y = (max_y - min_y) / (n_labels_ticks_y - 1)
                tick_y_lower = min_y
                tick_y_higher = max_y
                ticks_y = np.arange(tick_y_lower, tick_y_higher + (delta_y / 2), delta_y)

            ax.set_yticks(ticks_y)
        if stagger_labels_ticks_y:
            labels_ticks_y = (
                [str(l) if not i % 2 else '\n' + str(l) for i, l in enumerate(labels_ticks_y)])
        ax.set_yticklabels(
            labels_ticks_y, fontsize=font_size_labels_ticks_y,
            rotation=rotation_labels_ticks_y)

    if annot:

        if array_annot is None:
            array_annot = data

        # Normalize the threshold to the images color range.
        if colors_annot is None:
            colors_annot = ["black", "white"]
        elif isinstance(colors_annot, (list, tuple, np.ndarray)):
            pass
        else:
            colors_annot = [colors_annot]

        n_colors_annot = len(colors_annot)

        # kw = dict(horizontalalignment="center",
        #           verticalalignment="center",
        #           fontsize=font_size_annot,
        #           color=colors_annot[0])

        # Get the formatter in case a string is supplied
        # if isinstance(fmt_annot, str):
        #     fmt_annot = ticker.StrMethodFormatter(fmt_annot)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        # texts = []
        if n_colors_annot == 1:
            # Set default alignment to center, but allow it to be
            # overwritten by textkw.
            kw = dict(horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=font_size_annot,
                      color=colors_annot[0])

            for i in range(data.shape[dict_axes['y']]):
                for j in range(data.shape[dict_axes['x']]):

                    annot_i_j = fmt_annot.format(array_annot[i, j])

                    if annot_i_j[0] in ['-', '+']:
                        if annot_i_j[slice(1, 3, 1)] == '0.':
                            annot_i_j = annot_i_j[0] + annot_i_j[slice(2, None, 1)]
                    else:
                        if annot_i_j[slice(0, 2, 1)] == '0.':
                            annot_i_j = annot_i_j[slice(1, None, 1)]

                    im.axes.text(j, i, annot_i_j, **kw)

        elif n_colors_annot == 2:
            # Set default alignment to center, but allow it to be
            # overwritten by textkw.
            kw = dict(horizontalalignment="center",
                      verticalalignment="center",
                      fontsize=font_size_annot)

            if threshold_change_color_annot is not None:
                threshold = im.norm(threshold_change_color_annot)
            else:
                threshold = im.norm(data[cp_maths.is_not_nan(data)].max()) / 2

            for i in range(data.shape[dict_axes['y']]):
                for j in range(data.shape[dict_axes['x']]):

                    if cp_maths.is_not_nan(data[i, j]):

                        kw.update(color=colors_annot[int(im.norm(data[i, j]) > threshold)])

                        annot_i_j = fmt_annot.format(array_annot[i, j])

                        if annot_i_j[0] in ['-', '+']:
                            if annot_i_j[slice(1, 3, 1)] == '0.':
                                annot_i_j = annot_i_j[0] + annot_i_j[slice(2, None, 1)]
                        else:
                            if annot_i_j[slice(0, 2, 1)] == '0.':
                                annot_i_j = annot_i_j[slice(1, None, 1)]

                        im.axes.text(j, i, annot_i_j, **kw)
        else:
            raise ValueError('colors_annot')

    if tight_layout:
        plt.tight_layout()


def single_figure_multi_plots(
        data, dict_axes,
        annot=False, array_annot=None, fmts_annot='{:.2f}', colors_annot=None,
        font_sizes_annot=None, thresholds_change_color_annot=None,
        cmaps=None, n_levels_cmaps=None,
        add_cbars=True, labels_cbars=None, font_sizes_labels_cbars=None,
        labels_ticks_cbars=None, ticks_cbars=None, n_ticks_cbars=None,
        font_sizes_labels_ticks_cbars=None, rotations_labels_ticks_cbars=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_cmaps=None, maxes_cmaps=None,
        id_figure=None, share_x='none', share_y='none',
        h_space=None, w_space=None,
        add_letters_to_titles=True, letter_start_titles=None, template_letter_addition_titles=None,
        ratio_plot=None, maximum_n_pixels_per_plot=500, tight_layout=True):

    figures_existing = plt.get_fignums()
    n_figures_existing = len(figures_existing)
    if id_figure is None:
        i = 0
        while id_figure is None:
            if i in figures_existing:
                pass
            else:
                id_figure = i
            i += 1
    else:
        if id_figure in figures_existing:
            print('Warning: overwriting figure {}.'.format(id_figure))

    # format axes
    keys_axes_expected = np.asarray(['rows', 'columns', 'y', 'x'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(dict_axes, dict):
        raise TypeError('The type of "dict_axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(dict_axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            dict_axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        dict_axes, keys_axes_expected, values_axes_expected, name_dictionary='dict_axes')
    cp_check.values_are_not_repeated(dict_axes, name_dictionary='dict_axes')

    if isinstance(data, np.ndarray):
        pass
    else:
        data = np.asarray(data)

    shape_data = np.asarray(data.shape, dtype='i')
    n_dim_data = shape_data.size
    if n_dim_data != n_keys_axes_expected:
        raise ValueError('data has to be a {}d array'.format(n_keys_axes_expected))

    axes_rc_in_data = np.asarray([dict_axes['rows'], dict_axes['columns']], dtype='i')
    axes_rc_in_data_sort = np.sort(axes_rc_in_data)
    n_axes_rc = axes_rc_in_data.size

    n_rows, n_columns = shape_data[axes_rc_in_data]
    n_subplots = n_rows * n_columns

    shape_rc = shape_data[axes_rc_in_data_sort]

    dict_axes_next = {}
    if dict_axes['y'] > dict_axes['x']:
        dict_axes_next['y'] = 1
        dict_axes_next['x'] = 0
    elif dict_axes['y'] < dict_axes['x']:
        dict_axes_next['y'] = 0
        dict_axes_next['x'] = 1

    if add_letters_to_titles:
        if letter_start_titles is None:
            letter_start_titles = 'A'

        num_letter_start_titles = ord(letter_start_titles)

        if template_letter_addition_titles is None:
            template_letter_addition_titles = '({subplot_letter:s})  '

        # # len_addition = len(addition)
        # if titles is None:
        #     titles = addition
        # elif isinstance(titles, str):
        #     titles = addition + titles
        # elif isinstance(titles, (np.ndarray, list, tuple)):
        #
        #     if isinstance(titles, (list, tuple)):
        #         titles = np.asarray(titles, dtype='U')
        #
        #     if titles.dtype.char != 'U':
        #         idx = np.empty(n_axes_rc, dtype='i')
        #         idx[:] = 0
        #         if titles[tuple(idx)] is None:
        #             titles = addition
        #         else:
        #             titles = np.char.add(addition, titles.astype('U'))
        #
        #     else:
        #         titles = np.char.add(addition, titles)
        # else:
        #     titles = np.char.add(addition, np.asarray(titles, dtype='U'))
    else:
        num_letter_start_titles = None

    dict_parameters_rc = dict(
        annot=annot, fmts_annot=fmts_annot, colors_annot=colors_annot, font_sizes_annot=font_sizes_annot,
        thresholds_change_color_annot=thresholds_change_color_annot,
        cmaps=cmaps, n_levels_cmaps=n_levels_cmaps,
        add_cbars=add_cbars, labels_cbars=labels_cbars, font_sizes_labels_cbars=font_sizes_labels_cbars,
        labels_ticks_cbars=labels_ticks_cbars, ticks_cbars=ticks_cbars, n_ticks_cbars=n_ticks_cbars,
        font_sizes_labels_ticks_cbars=font_sizes_labels_ticks_cbars,
        rotations_labels_ticks_cbars=rotations_labels_ticks_cbars,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_cmaps=mins_cmaps, maxes_cmaps=maxes_cmaps)

    dict_parameters_rc = cp_format.format_shape_arguments(dict_parameters_rc, shape_rc)

    annot = dict_parameters_rc['annot']
    fmts_annot = dict_parameters_rc['fmts_annot']
    colors_annot = dict_parameters_rc['colors_annot']
    font_sizes_annot = dict_parameters_rc['font_sizes_annot']
    thresholds_change_color_annot = dict_parameters_rc['thresholds_change_color_annot']
    cmaps = dict_parameters_rc['cmaps']
    n_levels_cmaps = dict_parameters_rc['n_levels_cmaps']
    add_cbars = dict_parameters_rc['add_cbars']
    labels_cbars = dict_parameters_rc['labels_cbars']
    font_sizes_labels_cbars = dict_parameters_rc['font_sizes_labels_cbars']
    labels_ticks_cbars = dict_parameters_rc['labels_ticks_cbars']
    ticks_cbars = dict_parameters_rc['ticks_cbars']
    n_ticks_cbars = dict_parameters_rc['n_ticks_cbars']
    font_sizes_labels_ticks_cbars = dict_parameters_rc['font_sizes_labels_ticks_cbars']
    rotations_labels_ticks_cbars = dict_parameters_rc['rotations_labels_ticks_cbars']
    titles = dict_parameters_rc['titles']
    font_sizes_titles = dict_parameters_rc['font_sizes_titles']
    rotations_titles = dict_parameters_rc['rotations_titles']
    labels_x = dict_parameters_rc['labels_x']
    font_sizes_labels_x = dict_parameters_rc['font_sizes_labels_x']
    rotations_labels_x = dict_parameters_rc['rotations_labels_x']
    labels_y = dict_parameters_rc['labels_y']
    font_sizes_labels_y = dict_parameters_rc['font_sizes_labels_y']
    rotations_labels_y = dict_parameters_rc['rotations_labels_y']
    labels_ticks_x = dict_parameters_rc['labels_ticks_x']
    ticks_x = dict_parameters_rc['ticks_x']
    n_ticks_x = dict_parameters_rc['n_ticks_x']
    stagger_labels_ticks_x = dict_parameters_rc['stagger_labels_ticks_x']
    font_sizes_labels_ticks_x = dict_parameters_rc['font_sizes_labels_ticks_x']
    rotations_labels_ticks_x = dict_parameters_rc['rotations_labels_ticks_x']
    labels_ticks_y = dict_parameters_rc['labels_ticks_y']
    ticks_y = dict_parameters_rc['ticks_y']
    n_ticks_y = dict_parameters_rc['n_ticks_y']
    stagger_labels_ticks_y = dict_parameters_rc['stagger_labels_ticks_y']
    font_sizes_labels_ticks_y = dict_parameters_rc['font_sizes_labels_ticks_y']
    rotations_labels_ticks_y = dict_parameters_rc['rotations_labels_ticks_y']
    mins_cmaps = dict_parameters_rc['mins_cmaps']
    maxes_cmaps = dict_parameters_rc['maxes_cmaps']

    if ratio_plot is None:
        ratio_plot = {}
        if data.shape[dict_axes['x']] > data.shape[dict_axes['y']]:
            ratio_plot['x'] = 1
            ratio_plot['y'] = data.shape[dict_axes['y']] / data.shape[dict_axes['x']]
        elif data.shape[dict_axes['x']] < data.shape[dict_axes['y']]:
            ratio_plot['x'] = data.shape[dict_axes['x']] / data.shape[dict_axes['y']]
            ratio_plot['y'] = 1
        else:
            ratio_plot['x'] = 1
            ratio_plot['y'] = 1

    fig, ax = plt.subplots(
        n_rows, n_columns, sharex=share_x, sharey=share_y, squeeze=False,
        num=id_figure, frameon=False, dpi=my_dpi,
        figsize=(((maximum_n_pixels_per_plot * ratio_plot['x']) * n_columns) / my_dpi,
                 ((maximum_n_pixels_per_plot * ratio_plot['y']) * n_rows) / my_dpi))

    indexes_rcyx_i = np.empty(n_keys_axes_expected, dtype='O')
    for d in [dict_axes['y'], dict_axes['x']]:
        indexes_rcyx_i[d] = slice(0, shape_data[d], 1)

    indexes_rc_i = np.empty(n_axes_rc, dtype='O')

    if dict_axes['rows'] < dict_axes['columns']:
        indexes_combinations_rc = slice(0, n_axes_rc, 1)
    elif dict_axes['rows'] > dict_axes['columns']:
        indexes_combinations_rc = slice(-1, -(n_axes_rc + 1), -1)
    else:
        raise ValueError('axes')

    array_annot_i = None

    i = 0
    for combination_rc_i in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_rc):

        ax[tuple(combination_rc_i[indexes_combinations_rc])].tick_params(axis='both', labelbottom=True, labelleft=True)

        indexes_rcyx_i[axes_rc_in_data_sort] = combination_rc_i
        tuple_indexes_rcyx_i = tuple(indexes_rcyx_i)

        indexes_rc_i[slice(0, n_axes_rc, 1)] = combination_rc_i
        tuple_indexes_rc_i = tuple(indexes_rc_i)

        if array_annot is not None:
            array_annot_i = array_annot[tuple_indexes_rcyx_i]

        if add_letters_to_titles:
            title_i = template_letter_addition_titles.format(subplot_letter=chr(num_letter_start_titles + i))
            if titles[tuple_indexes_rc_i] is not None:
                title_i += titles[tuple_indexes_rc_i].tolist()
        else:
            title_i = titles[tuple_indexes_rc_i]
            if isinstance(title_i, (np.ndarray)):
                title_i = title_i.tolist()

        single_figure_single_plot(
            data[tuple_indexes_rcyx_i], dict_axes=dict_axes_next,
            annot=annot[tuple_indexes_rc_i], array_annot=array_annot_i, fmt_annot=fmts_annot[tuple_indexes_rc_i],
            colors_annot=colors_annot[tuple_indexes_rc_i],
            font_size_annot=font_sizes_annot[tuple_indexes_rc_i],
            threshold_change_color_annot=thresholds_change_color_annot[tuple_indexes_rc_i],
            cmap=cmaps[tuple_indexes_rc_i], n_levels_cmap=n_levels_cmaps[tuple_indexes_rc_i],
            add_cbar=add_cbars[tuple_indexes_rc_i], label_cbar=labels_cbars[tuple_indexes_rc_i],
            font_size_label_cbar=font_sizes_labels_cbars[tuple_indexes_rc_i],
            labels_ticks_cbar=labels_ticks_cbars[tuple_indexes_rc_i],
            ticks_cbar=ticks_cbars[tuple_indexes_rc_i], n_ticks_cbar=n_ticks_cbars[tuple_indexes_rc_i],
            font_size_labels_ticks_cbar=font_sizes_labels_ticks_cbars[tuple_indexes_rc_i],
            rotation_labels_ticks_cbar=rotations_labels_ticks_cbars[tuple_indexes_rc_i],
            title=title_i, font_size_title=font_sizes_titles[tuple_indexes_rc_i],
            rotation_title=rotations_titles[tuple_indexes_rc_i],
            label_x=labels_x[tuple_indexes_rc_i], font_size_label_x=font_sizes_labels_x[tuple_indexes_rc_i],
            rotation_label_x=rotations_labels_x[tuple_indexes_rc_i],
            label_y=labels_y[tuple_indexes_rc_i], font_size_label_y=font_sizes_labels_y[tuple_indexes_rc_i],
            rotation_label_y=rotations_labels_y[tuple_indexes_rc_i],
            labels_ticks_x=labels_ticks_x[tuple_indexes_rc_i], ticks_x=ticks_x[tuple_indexes_rc_i],
            n_ticks_x=n_ticks_x[tuple_indexes_rc_i],
            stagger_labels_ticks_x=stagger_labels_ticks_x[tuple_indexes_rc_i],
            font_size_labels_ticks_x=font_sizes_labels_ticks_x[tuple_indexes_rc_i],
            rotation_labels_ticks_x=rotations_labels_ticks_x[tuple_indexes_rc_i],
            labels_ticks_y=labels_ticks_y[tuple_indexes_rc_i], ticks_y=ticks_y[tuple_indexes_rc_i],
            n_ticks_y=n_ticks_y[tuple_indexes_rc_i],
            stagger_labels_ticks_y=stagger_labels_ticks_y[tuple_indexes_rc_i],
            font_size_labels_ticks_y=font_sizes_labels_ticks_y[tuple_indexes_rc_i],
            rotation_labels_ticks_y=rotations_labels_ticks_y[tuple_indexes_rc_i],
            min_cmap=mins_cmaps[tuple_indexes_rc_i], max_cmap=maxes_cmaps[tuple_indexes_rc_i],
            ax=ax[tuple(combination_rc_i[indexes_combinations_rc])], tight_layout=False)

        i += 1

    if tight_layout:
        plt.tight_layout()

    if any([h_space is not None, w_space is not None]):
        plt.subplots_adjust(hspace=h_space, wspace=w_space)


def multi_figures_single_plot(
        data, dict_axes,
        annot=False, array_annot=None, fmts_annot='{:.2f}', colors_annot=None,
        font_sizes_annot=None, thresholds_change_color_annot=None,
        cmaps=None, n_levels_cmaps=None,
        add_cbars=True, labels_cbars=None, font_sizes_labels_cbars=None,
        labels_ticks_cbars=None, ticks_cbars=None, n_ticks_cbars=None,
        font_sizes_labels_ticks_cbars=None, rotations_labels_ticks_cbars=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_cmaps=None, maxes_cmaps=None,
        id_figures=None, ratios_plot=None, maximum_n_pixels_per_plot=500, tight_layouts=True):

    # format axes
    keys_axes_expected = np.asarray(['figures', 'y', 'x'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(dict_axes, dict):
        raise TypeError('The type of "dict_axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(dict_axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            dict_axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        dict_axes, keys_axes_expected, values_axes_expected, name_dictionary='dict_axes')
    cp_check.values_are_not_repeated(dict_axes, name_dictionary='dict_axes')

    if isinstance(data, np.ndarray):
        pass
    else:
        data = np.asarray(data)

    shape_data = np.asarray(data.shape, dtype='i')
    n_dim_data = shape_data.size
    if n_dim_data != n_keys_axes_expected:
        raise ValueError('data has to be a {}d array'.format(n_keys_axes_expected))

    n_figures = shape_data[dict_axes['figures']]

    figures_existing = plt.get_fignums()
    n_figures_existing = len(figures_existing)
    if id_figures is None:
        id_figures = [None for f in range(0, n_figures, 1)]  # type: list
        i = 0
        f = 0
        while f < n_figures:
            if i in figures_existing:
                pass
            else:
                id_figures[f] = i
                f += 1
            i += 1
    else:
        for f in id_figures:
            if f in figures_existing:
                print('Warning: overwriting figure {}.'.format(f))

    dict_axes_next = {}
    for k in dict_axes:
        if k == 'figures':
            continue
        if dict_axes[k] < dict_axes['figures']:
            dict_axes_next[k] = dict_axes[k]
        elif dict_axes[k] > dict_axes['figures']:
            dict_axes_next[k] = dict_axes[k] - 1
        else:
            raise ValueError('\n\tThe following condition is not met:\n'
                             '\t\tdict_axes[\'{}\'] \u2260 dict_axes[\'figures\']'.format(k))

    dict_parameters_f = dict(
        annot=annot, fmts_annot=fmts_annot, colors_annot=colors_annot,
        font_sizes_annot=font_sizes_annot, thresholds_change_color_annot=thresholds_change_color_annot,
        cmaps=cmaps, n_levels_cmaps=n_levels_cmaps,
        add_cbars=add_cbars, labels_cbars=labels_cbars, font_sizes_labels_cbars=font_sizes_labels_cbars,
        labels_ticks_cbars=labels_ticks_cbars, ticks_cbars=ticks_cbars, n_ticks_cbars=n_ticks_cbars,
        font_sizes_labels_ticks_cbars=font_sizes_labels_ticks_cbars,
        rotations_labels_ticks_cbars=rotations_labels_ticks_cbars,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_cmaps=mins_cmaps, maxes_cmaps=maxes_cmaps,
        ratios_plot=ratios_plot, maximum_n_pixels_per_plot=maximum_n_pixels_per_plot,
        tight_layouts=tight_layouts)

    dict_parameters_f = cp_format.format_shape_arguments(dict_parameters_f, n_figures)

    annot = dict_parameters_f['annot']
    fmts_annot = dict_parameters_f['fmts_annot']
    colors_annot = dict_parameters_f['colors_annot']
    font_sizes_annot = dict_parameters_f['font_sizes_annot']
    thresholds_change_color_annot = dict_parameters_f['thresholds_change_color_annot']
    cmaps = dict_parameters_f['cmaps']
    n_levels_cmaps = dict_parameters_f['n_levels_cmaps']
    add_cbars = dict_parameters_f['add_cbars']
    labels_cbars = dict_parameters_f['labels_cbars']
    font_sizes_labels_cbars = dict_parameters_f['font_sizes_labels_cbars']
    labels_ticks_cbars = dict_parameters_f['labels_ticks_cbars']
    ticks_cbars = dict_parameters_f['ticks_cbars']
    n_ticks_cbars = dict_parameters_f['n_ticks_cbars']
    font_sizes_labels_ticks_cbars = dict_parameters_f['font_sizes_labels_ticks_cbars']
    rotations_labels_ticks_cbars = dict_parameters_f['rotations_labels_ticks_cbars']
    titles = dict_parameters_f['titles']
    font_sizes_titles = dict_parameters_f['font_sizes_titles']
    rotations_titles = dict_parameters_f['rotations_titles']
    labels_x = dict_parameters_f['labels_x']
    font_sizes_labels_x = dict_parameters_f['font_sizes_labels_x']
    rotations_labels_x = dict_parameters_f['rotations_labels_x']
    labels_y = dict_parameters_f['labels_y']
    font_sizes_labels_y = dict_parameters_f['font_sizes_labels_y']
    rotations_labels_y = dict_parameters_f['rotations_labels_y']
    labels_ticks_x = dict_parameters_f['labels_ticks_x']
    ticks_x = dict_parameters_f['ticks_x']
    n_ticks_x = dict_parameters_f['n_ticks_x']
    stagger_labels_ticks_x = dict_parameters_f['stagger_labels_ticks_x']
    font_sizes_labels_ticks_x = dict_parameters_f['font_sizes_labels_ticks_x']
    rotations_labels_ticks_x = dict_parameters_f['rotations_labels_ticks_x']
    labels_ticks_y = dict_parameters_f['labels_ticks_y']
    ticks_y = dict_parameters_f['ticks_y']
    n_ticks_y = dict_parameters_f['n_ticks_y']
    stagger_labels_ticks_y = dict_parameters_f['stagger_labels_ticks_y']
    font_sizes_labels_ticks_y = dict_parameters_f['font_sizes_labels_ticks_y']
    rotations_labels_ticks_y = dict_parameters_f['rotations_labels_ticks_y']
    mins_cmaps = dict_parameters_f['mins_cmaps']
    maxes_cmaps = dict_parameters_f['maxes_cmaps']
    ratios_plot = dict_parameters_f['ratios_plot']
    maximum_n_pixels_per_plot = dict_parameters_f['maximum_n_pixels_per_plot']
    tight_layouts = dict_parameters_f['tight_layouts']

    indexes_fyx_f = np.asarray([slice(0, shape_data[d], 1) for d in range(0, n_keys_axes_expected, 1)], dtype='O')

    array_annot_f = None
    for f in range(n_figures):

        if ratios_plot[f] is None:
            ratio_plot_f = {}
            if data.shape[dict_axes['x']] > data.shape[dict_axes['y']]:
                ratio_plot_f['x'] = 1
                ratio_plot_f['y'] = data.shape[dict_axes['y']] / data.shape[dict_axes['x']]
            elif data.shape[dict_axes['x']] < data.shape[dict_axes['y']]:
                ratio_plot_f['x'] = data.shape[dict_axes['x']] / data.shape[dict_axes['y']]
                ratio_plot_f['y'] = 1
            else:
                ratio_plot_f['x'] = 1
                ratio_plot_f['y'] = 1
        else:
            ratio_plot_f = ratios_plot[f]

        fig_f = plt.figure(
            num=id_figures[f], frameon=False, dpi=my_dpi,
            figsize=((maximum_n_pixels_per_plot[f] * ratio_plot_f['x']) / my_dpi, (maximum_n_pixels_per_plot[f] * ratio_plot_f['y']) / my_dpi))

        ax_f = plt.subplot(1, 1, 1)

        # ax_f.tick_params(axis='both', labelbottom=True, labelleft=True)

        indexes_fyx_f[dict_axes['figures']] = f
        tuple_indexes_fyx_f = tuple(indexes_fyx_f)

        if array_annot is not None:
            array_annot_f = array_annot[tuple_indexes_fyx_f]

        single_figure_single_plot(
            data[tuple_indexes_fyx_f], dict_axes=dict_axes_next,
            annot=annot[f], array_annot=array_annot_f, fmt_annot=fmts_annot[f], colors_annot=colors_annot[f],
            font_size_annot=font_sizes_annot[f], threshold_change_color_annot=thresholds_change_color_annot[f],
            cmap=cmaps[f], n_levels_cmap=n_levels_cmaps[f],
            add_cbar=add_cbars[f], label_cbar=labels_cbars[f], font_size_label_cbar=font_sizes_labels_cbars[f],
            labels_ticks_cbar=labels_ticks_cbars[f], ticks_cbar=ticks_cbars[f], n_ticks_cbar=n_ticks_cbars[f],
            font_size_labels_ticks_cbar=font_sizes_labels_ticks_cbars[f],
            rotation_labels_ticks_cbar=rotations_labels_ticks_cbars[f],
            title=titles[f], font_size_title=font_sizes_titles[f], rotation_title=rotations_titles[f],
            label_x=labels_x[f], font_size_label_x=font_sizes_labels_x[f], rotation_label_x=rotations_labels_x[f],
            label_y=labels_y[f], font_size_label_y=font_sizes_labels_y[f], rotation_label_y=rotations_labels_y[f],
            labels_ticks_x=labels_ticks_x[f], ticks_x=ticks_x[f], n_ticks_x=n_ticks_x[f],
            stagger_labels_ticks_x=stagger_labels_ticks_x[f], font_size_labels_ticks_x=font_sizes_labels_ticks_x[f],
            rotation_labels_ticks_x=rotations_labels_ticks_x[f],
            labels_ticks_y=labels_ticks_y[f], ticks_y=ticks_y[f], n_ticks_y=n_ticks_y[f],
            stagger_labels_ticks_y=stagger_labels_ticks_y[f], font_size_labels_ticks_y=font_sizes_labels_ticks_y[f],
            rotation_labels_ticks_y=rotations_labels_ticks_y[f],
            min_cmap=mins_cmaps[f], max_cmap=maxes_cmaps[f],
            ax=ax_f, ratio_plot='not needed', maximum_n_pixels_per_plot='not needed', tight_layout=tight_layouts[f])


def multi_figures_multi_plots(
        data, dict_axes,
        annot=False, array_annot=None, fmts_annot='{:.2f}', colors_annot=None,
        font_sizes_annot=None, thresholds_change_color_annot=None,
        cmaps=None, n_levels_cmaps=None,
        add_cbars=True, labels_cbars=None, font_sizes_labels_cbars=None,
        labels_ticks_cbars=None, ticks_cbars=None, n_ticks_cbars=None,
        font_sizes_labels_ticks_cbars=None, rotations_labels_ticks_cbars=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_cmaps=None, maxes_cmaps=None,
        id_figures=None, share_x='none', share_y='none',
        h_spaces=None, w_spaces=None,
        add_letters_to_titles=True, letter_start_titles=None, template_letter_addition_titles=None,
        ratios_plot=None, maximum_n_pixels_per_plot=500, tight_layouts=True):

    # format axes
    keys_axes_expected = np.asarray(['figures', 'rows', 'columns', 'y', 'x'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(dict_axes, dict):
        raise TypeError('The type of "dict_axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(dict_axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            dict_axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        dict_axes, keys_axes_expected, values_axes_expected, name_dictionary='dict_axes')
    cp_check.values_are_not_repeated(dict_axes, name_dictionary='dict_axes')

    if isinstance(data, np.ndarray):
        pass
    else:
        data = np.asarray(data)

    shape_data = np.asarray(data.shape, dtype='i')
    n_dim_data = shape_data.size
    if n_dim_data != n_keys_axes_expected:
        raise ValueError('data has to be a {}d array'.format(n_keys_axes_expected))

    axes_frc_in_data = np.asarray([dict_axes['figures'], dict_axes['rows'], dict_axes['columns']], dtype='i')
    axes_frc_in_data_sort = np.sort(axes_frc_in_data)
    n_axes_frc = axes_frc_in_data.size

    n_figures, n_rows, n_columns = shape_data[axes_frc_in_data]
    shape_frc = shape_data[axes_frc_in_data_sort]

    figures_existing = plt.get_fignums()
    n_figures_existing = len(figures_existing)
    if id_figures is None:
        id_figures = [None for f in range(0, n_figures, 1)]  # type: list
        i = 0
        f = 0
        while f < n_figures:
            if i in figures_existing:
                pass
            else:
                id_figures[f] = i
                f += 1
            i += 1
    else:
        for f in id_figures:
            if f in figures_existing:
                print('Warning: overwriting figure {}.'.format(f))

    dict_axes_next = {}
    for k in dict_axes:
        if k == 'figures':
            continue
        if dict_axes[k] < dict_axes['figures']:
            dict_axes_next[k] = dict_axes[k]
        elif dict_axes[k] > dict_axes['figures']:
            dict_axes_next[k] = dict_axes[k] - 1
        else:
            raise ValueError('\n\tThe following condition is not met:\n'
                             '\t\tdict_axes[\'{}\'] \u2260 dict_axes[\'figures\']'.format(k))

    dict_parameters_frc = dict(
        annot=annot, fmts_annot=fmts_annot, colors_annot=colors_annot,
        font_sizes_annot=font_sizes_annot, thresholds_change_color_annot=thresholds_change_color_annot,
        cmaps=cmaps, n_levels_cmaps=n_levels_cmaps,
        add_cbars=add_cbars, labels_cbars=labels_cbars, font_sizes_labels_cbars=font_sizes_labels_cbars,
        labels_ticks_cbars=labels_ticks_cbars, ticks_cbars=ticks_cbars, n_ticks_cbars=n_ticks_cbars,
        font_sizes_labels_ticks_cbars=font_sizes_labels_ticks_cbars,
        rotations_labels_ticks_cbars=rotations_labels_ticks_cbars,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_cmaps=mins_cmaps, maxes_cmaps=maxes_cmaps)

    dict_parameters_frc = cp_format.format_shape_arguments(dict_parameters_frc, shape_frc)

    annot = dict_parameters_frc['annot']
    fmts_annot = dict_parameters_frc['fmts_annot']
    colors_annot = dict_parameters_frc['colors_annot']
    font_sizes_annot = dict_parameters_frc['font_sizes_annot']
    thresholds_change_color_annot = dict_parameters_frc['thresholds_change_color_annot']
    cmaps = dict_parameters_frc['cmaps']
    n_levels_cmaps = dict_parameters_frc['n_levels_cmaps']
    add_cbars = dict_parameters_frc['add_cbars']
    labels_cbars = dict_parameters_frc['labels_cbars']
    font_sizes_labels_cbars = dict_parameters_frc['font_sizes_labels_cbars']
    labels_ticks_cbars = dict_parameters_frc['labels_ticks_cbars']
    ticks_cbars = dict_parameters_frc['ticks_cbars']
    n_ticks_cbars = dict_parameters_frc['n_ticks_cbars']
    font_sizes_labels_ticks_cbars = dict_parameters_frc['font_sizes_labels_ticks_cbars']
    rotations_labels_ticks_cbars = dict_parameters_frc['rotations_labels_ticks_cbars']
    titles = dict_parameters_frc['titles']
    font_sizes_titles = dict_parameters_frc['font_sizes_titles']
    rotations_titles = dict_parameters_frc['rotations_titles']
    labels_x = dict_parameters_frc['labels_x']
    font_sizes_labels_x = dict_parameters_frc['font_sizes_labels_x']
    rotations_labels_x = dict_parameters_frc['rotations_labels_x']
    labels_y = dict_parameters_frc['labels_y']
    font_sizes_labels_y = dict_parameters_frc['font_sizes_labels_y']
    rotations_labels_y = dict_parameters_frc['rotations_labels_y']
    labels_ticks_x = dict_parameters_frc['labels_ticks_x']
    ticks_x = dict_parameters_frc['ticks_x']
    n_ticks_x = dict_parameters_frc['n_ticks_x']
    stagger_labels_ticks_x = dict_parameters_frc['stagger_labels_ticks_x']
    font_sizes_labels_ticks_x = dict_parameters_frc['font_sizes_labels_ticks_x']
    rotations_labels_ticks_x = dict_parameters_frc['rotations_labels_ticks_x']
    labels_ticks_y = dict_parameters_frc['labels_ticks_y']
    ticks_y = dict_parameters_frc['ticks_y']
    n_ticks_y = dict_parameters_frc['n_ticks_y']
    stagger_labels_ticks_y = dict_parameters_frc['stagger_labels_ticks_y']
    font_sizes_labels_ticks_y = dict_parameters_frc['font_sizes_labels_ticks_y']
    rotations_labels_ticks_y = dict_parameters_frc['rotations_labels_ticks_y']
    mins_cmaps = dict_parameters_frc['mins_cmaps']
    maxes_cmaps = dict_parameters_frc['maxes_cmaps']

    dict_parameters_f = dict(
        share_x=share_x, share_y=share_y,
        h_spaces=h_spaces, w_spaces=w_spaces, add_letters_to_titles=add_letters_to_titles,
        ratios_plot=ratios_plot, maximum_n_pixels_per_plot=maximum_n_pixels_per_plot,
        tight_layouts=tight_layouts)

    dict_parameters_f = cp_format.format_shape_arguments(dict_parameters_f, [n_figures])

    share_x = dict_parameters_f['share_x']
    share_y = dict_parameters_f['share_y']
    h_spaces = dict_parameters_f['h_spaces']
    w_spaces = dict_parameters_f['w_spaces']
    add_letters_to_titles = dict_parameters_f['add_letters_to_titles']
    ratios_plot = dict_parameters_f['ratios_plot']
    maximum_n_pixels_per_plot = dict_parameters_f['maximum_n_pixels_per_plot']
    tight_layouts = dict_parameters_f['tight_layouts']

    indexes_frcyx_f = np.asarray([slice(0, shape_data[d], 1) for d in range(0, n_keys_axes_expected, 1)], dtype='O')

    indexes_frc_f = np.asarray([slice(0, shape_frc[d], 1) for d in range(0, n_axes_frc, 1)], dtype='O')

    axis_f_in_parms_frc = 0
    if dict_axes['figures'] > dict_axes['rows']:
        axis_f_in_parms_frc += 1
    if dict_axes['figures'] > dict_axes['columns']:
        axis_f_in_parms_frc += 1

    array_annot_f = None
    for f in range(n_figures):

        indexes_frcyx_f[dict_axes['figures']] = f
        tuple_indexes_frcyx_f = tuple(indexes_frcyx_f)

        indexes_frc_f[axis_f_in_parms_frc] = f
        tuple_indexes_frc_f = tuple(indexes_frc_f)

        if array_annot is not None:
            array_annot_f = array_annot[tuple_indexes_frcyx_f]

        single_figure_multi_plots(
            data[tuple_indexes_frcyx_f], dict_axes=dict_axes_next,
            annot=annot[tuple_indexes_frc_f], array_annot=array_annot_f, fmts_annot=fmts_annot[tuple_indexes_frc_f],
            colors_annot=colors_annot[tuple_indexes_frc_f], font_sizes_annot=font_sizes_annot[tuple_indexes_frc_f],
            thresholds_change_color_annot=thresholds_change_color_annot[tuple_indexes_frc_f],
            cmaps=cmaps[tuple_indexes_frc_f], n_levels_cmaps=n_levels_cmaps[tuple_indexes_frc_f],
            add_cbars=add_cbars[tuple_indexes_frc_f], labels_cbars=labels_cbars[tuple_indexes_frc_f],
            font_sizes_labels_cbars=font_sizes_labels_cbars[tuple_indexes_frc_f],
            labels_ticks_cbars=labels_ticks_cbars[tuple_indexes_frc_f],
            ticks_cbars=ticks_cbars[tuple_indexes_frc_f], n_ticks_cbars=n_ticks_cbars[tuple_indexes_frc_f],
            font_sizes_labels_ticks_cbars=font_sizes_labels_ticks_cbars[tuple_indexes_frc_f],
            rotations_labels_ticks_cbars=rotations_labels_ticks_cbars[tuple_indexes_frc_f],
            titles=titles[tuple_indexes_frc_f], font_sizes_titles=font_sizes_titles[tuple_indexes_frc_f],
            rotations_titles=rotations_titles[tuple_indexes_frc_f],
            labels_x=labels_x[tuple_indexes_frc_f], font_sizes_labels_x=font_sizes_labels_x[tuple_indexes_frc_f],
            rotations_labels_x=rotations_labels_x[tuple_indexes_frc_f],
            labels_y=labels_y[tuple_indexes_frc_f], font_sizes_labels_y=font_sizes_labels_y[tuple_indexes_frc_f],
            rotations_labels_y=rotations_labels_y[tuple_indexes_frc_f],
            labels_ticks_x=labels_ticks_x[tuple_indexes_frc_f], ticks_x=ticks_x[tuple_indexes_frc_f],
            n_ticks_x=n_ticks_x[tuple_indexes_frc_f],
            stagger_labels_ticks_x=stagger_labels_ticks_x[tuple_indexes_frc_f],
            font_sizes_labels_ticks_x=font_sizes_labels_ticks_x[tuple_indexes_frc_f],
            rotations_labels_ticks_x=rotations_labels_ticks_x[tuple_indexes_frc_f],
            labels_ticks_y=labels_ticks_y[tuple_indexes_frc_f], ticks_y=ticks_y[tuple_indexes_frc_f],
            n_ticks_y=n_ticks_y[tuple_indexes_frc_f],
            stagger_labels_ticks_y=stagger_labels_ticks_y[tuple_indexes_frc_f],
            font_sizes_labels_ticks_y=font_sizes_labels_ticks_y[tuple_indexes_frc_f],
            rotations_labels_ticks_y=rotations_labels_ticks_y[tuple_indexes_frc_f],
            mins_cmaps=mins_cmaps[tuple_indexes_frc_f], maxes_cmaps=maxes_cmaps[tuple_indexes_frc_f],
            id_figure=id_figures[f], share_x=share_x[f], share_y=share_y[f],
            h_space=h_spaces[f], w_space=w_spaces[f],
            add_letters_to_titles=add_letters_to_titles[f], letter_start_titles=letter_start_titles,
            template_letter_addition_titles=template_letter_addition_titles,
            ratio_plot=ratios_plot[f], maximum_n_pixels_per_plot=maximum_n_pixels_per_plot[f],
            tight_layout=tight_layouts[f])
