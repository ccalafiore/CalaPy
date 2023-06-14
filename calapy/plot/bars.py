import numpy as np
from . import plt
from .parameters import my_dpi
from .. import combinations as cp_combinations
from .. import format as cp_format
from .. import check as cp_check


# todo: 1) add option save_figs to save the figures as they are made

# todo: 2) add option close_figs to close and delete the figures as they are saved

# todo: 3) add multi directories

# todo: 4) add flexible axis of size of 2 for asymmetric error_bars_x and error_bars_y

# todo: 5) change letter of formats from f to j

# todo: 6) if axes are shared, then chose whether to draw tick labels:
#  this can be done here ".tick_params(axis='both', labelbottom=True, labelleft=True)"

# todo: 7) fix bug of min_x, min_y, max_x, max_y


def single_figure_single_plot_single_format(
        x, y,
        width_bars=0.6, bottom_bars=None, align_bars='center', color_bars=None,
        width_bar_edges=None, color_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        color_error_bars=None, line_width_error_bars=None,
        size_caps=0.0, thickness_caps=None,
        title=None, font_size_title=None, rotation_title=None,
        label_x=None, font_size_label_x=None, rotation_label_x=None,
        label_y=None, font_size_label_y=None, rotation_label_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_size_labels_ticks_x=None, rotation_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_size_labels_ticks_y=None, rotation_labels_ticks_y=None,
        min_x=None, max_x=None, min_y=None, max_y=None,
        legend=False, label_legend=None, location_legend='best', font_size_legend=None, n_columns_legend=1,
        ax=None, n_pixels_x=300, n_pixels_y=300, log=False, tight_layout=True):

    # todo: """ """
    """"""

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
        fig = plt.figure(
            id_figure, frameon=False, dpi=my_dpi,
            figsize=(n_pixels_x / my_dpi, n_pixels_y / my_dpi))
        ax = plt.gca()
    else:
        fig = ax.figure

    error_kw = dict(elinewidth=line_width_error_bars, capthick=thickness_caps)

    ax.bar(
        x, y,
        width=width_bars, bottom=bottom_bars, align=align_bars, color=color_bars,
        edgecolor=color_bar_edges, linewidth=width_bar_edges,
        xerr=error_bars_x, yerr=error_bars_y,
        ecolor=color_error_bars, capsize=size_caps, error_kw=error_kw,
        log=log, label=label_legend)

    set_limits_x = (min_x is not None) or (max_x is not None)
    if set_limits_x:
        ax.set_xlim(xmin=min_x, xmax=max_x)

    set_limits_y = (min_y is not None) or (max_y is not None)
    if set_limits_y:
        ax.set_ylim(ymin=min_y, ymax=max_y)

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
                if min_x is None:
                    if isinstance(x, np.ndarray):
                        min_x = x.min()
                    else:
                        min_x = min(x)
                if max_x is None:
                    if isinstance(x, np.ndarray):
                        max_x = x.max()
                    else:
                        max_x = max(x)
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
            tmp_labels_ticks_x = ax.get_xticklabels()#[1:-1:1]
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
                if min_x is None:
                    if isinstance(x, np.ndarray):
                        min_x = x.min()
                    else:
                        min_x = min(x)
                if max_x is None:
                    if isinstance(x, np.ndarray):
                        max_x = x.max()
                    else:
                        max_x = max(x)
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
                if min_y is None:
                    if isinstance(y, np.ndarray):
                        min_y = y.min()
                    else:
                        min_y = min(y)
                if max_y is None:
                    if isinstance(y, np.ndarray):
                        max_y = y.max()
                    else:
                        max_y = max(y)
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
            tmp_labels_ticks_y = ax.get_yticklabels()#[1:-1:1]
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
                if min_y is None:
                    if isinstance(y, np.ndarray):
                        min_y = y.min()
                    else:
                        min_y = min(y)
                if max_y is None:
                    if isinstance(y, np.ndarray):
                        max_y = y.max()
                    else:
                        max_y = max(y)
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

    if legend:
        if isinstance(location_legend, np.ndarray):
            location_legend_form = location_legend.tolist()
        else:
            location_legend_form = location_legend
        ax.legend(loc=location_legend_form, fontsize=font_size_legend, ncols=n_columns_legend)

    if tight_layout:
        plt.tight_layout()


def single_figure_single_plot_multi_formats(
        x, y, axes,
        width_all_formats=0.8, width_between_formats=0.1,
        widths_bars=None, bottom_bars=None, align_bars='center', colors_bars=None,
        widths_bar_edges=None, colors_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        colors_error_bars=None, line_widths_error_bars=None,
        sizes_caps=0.0, thicknesses_caps=None,
        title=None, font_size_title=None, rotation_title=None,
        label_x=None, font_size_label_x=None, rotation_label_x=None,
        label_y=None, font_size_label_y=None, rotation_label_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_size_labels_ticks_x=None, rotation_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_size_labels_ticks_y=None, rotation_labels_ticks_y=None,
        min_x=None, max_x=None, min_y=None, max_y=None,
        legend=False, labels_legend=None, location_legend='best', font_size_legend=None, n_columns_legend=1,
        ax=None, n_pixels_x=300, n_pixels_y=300, log=False, tight_layout=True):

    # todo: """ """
    """"""

    # format axes
    keys_axes_expected = np.asarray(['formats', 'bars'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(axes, dict):
        raise TypeError('The type of "axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        axes, keys_axes_expected, values_axes_expected, name_dictionary='axes')
    cp_check.values_are_not_repeated(axes, name_dictionary='axes')

    if axes.get('formats') is None:
        if axes.get('bars') in values_axes_expected:
            axes['formats'] = int(axes['bars'] == 0)
        else:
            raise ValueError('axes')
    elif axes.get('bars') is None:
        if axes.get('formats') in values_axes_expected:
            axes['bars'] = int(axes['formats'] == 0)
        else:
            raise ValueError('axes')

    # format x and y

    if isinstance(x, np.ndarray):
        x = x
    else:
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = np.asarray(y)

    shape_x = np.asarray(x.shape, dtype='i')
    n_dim_x = shape_x.size
    shape_y = np.asarray(y.shape, dtype='i')
    n_dim_y = shape_y.size

    indexes_jp_j = np.empty(n_keys_axes_expected, dtype='O')

    if n_dim_y == n_keys_axes_expected:
        n_formats = shape_y[axes['formats']]

        if n_dim_x == n_keys_axes_expected:
            pass
        elif n_dim_x == 1:
            x = np.expand_dims(x, axis=axes['formats'])
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_y):
                    indexes_jp_j[d] = slice(0, shape_y[d], 1)
                x_tmp = x
                x = np.empty(shape=shape_y, dtype=x.dtype)
                x[tuple(indexes_jp_j)] = x_tmp
                shape_x = np.asarray(x.shape, dtype='i')
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))

    elif n_dim_y == 1:

        if n_dim_x == n_keys_axes_expected:
            n_formats = shape_x[axes['formats']]
            y = np.expand_dims(y, axis=axes['formats'])
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_x):
                    indexes_jp_j[d] = slice(0, shape_x[d], 1)
                y_tmp = y
                y = np.empty(shape=shape_x, dtype=y.dtype)
                y[tuple(indexes_jp_j)] = y_tmp
                shape_y = np.asarray(y.shape, dtype='i')

        elif n_dim_x == 1:
            n_formats = 1
            y = np.expand_dims(y, axis=axes['formats'])
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            x = np.expand_dims(x, axis=axes['formats'])
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))
    else:
        raise ValueError('y has to be a 1d or {}d array'.format(n_keys_axes_expected))

    # format error_bars_x
    if error_bars_x is None:
        symmetric_error_bars_x = True
        indexes_error_bars_x = None

    elif not np.iterable(error_bars_x):
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
        error_bars_x = [error_bars_x] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_x, np.ndarray):
            error_bars_x = np.asarray(error_bars_x)

        shape_error_bars_x_tmp = np.asarray(error_bars_x.shape, dtype='i')
        n_dim_error_bars_x = shape_error_bars_x_tmp.size
        indexes_error_bars_x = np.empty(n_dim_error_bars_x, dtype='O')

        if n_dim_error_bars_x == n_keys_axes_expected:
            symmetric_error_bars_x = True
            shape_error_bars_x = shape_y

        elif n_dim_error_bars_x == (n_keys_axes_expected + 1):
            symmetric_error_bars_x = False
            shape_error_bars_x = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_x')

        for d in range(n_dim_error_bars_x):
            indexes_error_bars_x[d] = slice(0, shape_error_bars_x[d], 1)

        if not np.all(shape_error_bars_x_tmp == shape_error_bars_x):
            error_bars_x_tmp = error_bars_x
            error_bars_x = np.empty(shape=shape_error_bars_x, dtype=error_bars_x.dtype)
            error_bars_x[tuple(indexes_error_bars_x)] = error_bars_x_tmp

    # format error_bars_y
    if error_bars_y is None:
        symmetric_error_bars_y = True
        indexes_error_bars_y = None

    elif not np.iterable(error_bars_y):
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
        error_bars_y = [error_bars_y] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_y, np.ndarray):
            error_bars_y = np.asarray(error_bars_y)

        shape_error_bars_y_tmp = np.asarray(error_bars_y.shape, dtype='i')
        n_dim_error_bars_y = shape_error_bars_y_tmp.size
        indexes_error_bars_y = np.empty(n_dim_error_bars_y, dtype='O')

        if n_dim_error_bars_y == n_keys_axes_expected:
            symmetric_error_bars_y = True
            shape_error_bars_y = shape_y

        elif n_dim_error_bars_y == (n_keys_axes_expected + 1):
            symmetric_error_bars_y = False
            shape_error_bars_y = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_y')

        for d in range(n_dim_error_bars_y):
            indexes_error_bars_y[d] = slice(0, shape_error_bars_y[d], 1)

        if not np.all(shape_error_bars_y_tmp == shape_error_bars_y):
            error_bars_y_tmp = error_bars_y
            error_bars_y = np.empty(shape=shape_error_bars_y, dtype=error_bars_y.dtype)
            error_bars_y[tuple(indexes_error_bars_y)] = error_bars_y_tmp

    # find id_figure
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
        fig = plt.figure(
            id_figure, frameon=False, dpi=my_dpi,
            figsize=(n_pixels_x / my_dpi, n_pixels_y / my_dpi))
        ax = plt.gca()
    else:
        fig = ax.figure

    if widths_bars is None:
        widths_bars = np.empty(n_formats, dtype='f')
        widths_bars[slice(0, n_formats, 1)] = (
                (width_all_formats / n_formats) - ((width_between_formats * (n_formats - 1)) / n_formats))
    else:
        try:
            width_all_formats = sum(widths_bars) + (width_between_formats * (n_formats - 1))
            if not isinstance(widths_bars, np.ndarray):
                widths_bars = np.asarray(widths_bars)
        except TypeError:
            widths_bars = np.asarray([widths_bars] * n_formats)
            width_all_formats = sum(widths_bars) + (width_between_formats * (n_formats - 1))

    x_formats = np.empty(n_formats, dtype='f')
    x_formats[0] = 0
    for j in range(1, n_formats, 1):
        x_formats[j] = widths_bars[slice(0, j, 1)].sum() + (width_between_formats * j)

    if align_bars == 'center':
        x_formats = x_formats - x_formats.mean(axis=0)
    # elif align_bars == 'right':
    #     x_formats = x_formats - x_formats.max(axis=0)

    dict_parameters_j = dict(
        colors_bars=colors_bars, widths_bar_edges=widths_bar_edges, colors_bar_edges=colors_bar_edges,
        colors_error_bars=colors_error_bars, line_widths_error_bars=line_widths_error_bars,
        sizes_caps=sizes_caps, thicknesses_caps=thicknesses_caps,
        labels_legend=labels_legend)

    dict_parameters_j = cp_format.format_shape_arguments(dict_parameters_j, n_formats)

    colors_bars = dict_parameters_j['colors_bars']
    widths_bar_edges = dict_parameters_j['widths_bar_edges']
    colors_bar_edges = dict_parameters_j['colors_bar_edges']
    colors_error_bars = dict_parameters_j['colors_error_bars']
    line_widths_error_bars = dict_parameters_j['line_widths_error_bars']
    sizes_caps = dict_parameters_j['sizes_caps']
    thicknesses_caps = dict_parameters_j['thicknesses_caps']
    labels_legend = dict_parameters_j['labels_legend']

    indexes_jp_j[axes['bars']] = slice(0, shape_y[axes['bars']], 1)

    for j in range(n_formats):
        indexes_jp_j[axes['formats']] = j

        if indexes_error_bars_x is None:
            error_bars_x_j = error_bars_x
        else:
            if symmetric_error_bars_x:
                indexes_error_bars_x[axes['formats']] = j
            else:
                indexes_error_bars_x[axes['formats'] + 1] = j
            error_bars_x_j = error_bars_x[tuple(indexes_error_bars_x)]

        if indexes_error_bars_y is None:
            error_bars_y_j = error_bars_y
        else:
            if symmetric_error_bars_y:
                indexes_error_bars_y[axes['formats']] = j
            else:
                indexes_error_bars_y[axes['formats'] + 1] = j
            error_bars_y_j = error_bars_y[tuple(indexes_error_bars_y)]

        tuple_indexes_jp_j = tuple(indexes_jp_j)

        # ax.bar(
        #     x[tuple_indexes_jp_j] + x_formats[j], y[tuple_indexes_jp_j],
        #     width=widths_bars[j], bottom=bottom_bars, align=align_bars, color=colors_bars[j],
        #     edgecolor=colors_bar_edges[j], linewidth=widths_bar_edges[j],
        #     xerr=error_bars_x_j, yerr=error_bars_y_j,
        #     ecolor=colors_error_bars[j], elinewidth=line_widths_error_bars[j],
        #     capsize=sizes_caps[j], capthick=thicknesses_caps[j],
        #     log=log, label=labels_legend[j])

        single_figure_single_plot_single_format(
            x[tuple_indexes_jp_j] + x_formats[j], y[tuple_indexes_jp_j],
            width_bars=widths_bars[j], bottom_bars=bottom_bars, align_bars=align_bars, color_bars=colors_bars[j],
            width_bar_edges=widths_bar_edges[j], color_bar_edges=colors_bar_edges[j],
            error_bars_x=error_bars_x_j, error_bars_y=error_bars_y_j,
            color_error_bars=colors_error_bars[j], line_width_error_bars=line_widths_error_bars[j],
            size_caps=sizes_caps[j], thickness_caps=thicknesses_caps[j],
            legend=False, label_legend=labels_legend[j],
            ax=ax, log=log, tight_layout=False)

    set_limits_x = (min_x is not None) or (max_x is not None)
    if set_limits_x:
        ax.set_xlim(xmin=min_x, xmax=max_x)

    set_limits_y = (min_y is not None) or (max_y is not None)
    if set_limits_y:
        ax.set_ylim(ymin=min_y, ymax=max_y)

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
                if min_x is None:
                    if isinstance(x, np.ndarray):
                        min_x = x.min()
                    else:
                        min_x = min(x)
                if max_x is None:
                    if isinstance(x, np.ndarray):
                        max_x = x.max()
                    else:
                        max_x = max(x)
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
            tmp_labels_ticks_x = ax.get_xticklabels()#[1:-1:1]
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
                if min_x is None:
                    if isinstance(x, np.ndarray):
                        min_x = x.min()
                    else:
                        min_x = min(x)
                if max_x is None:
                    if isinstance(x, np.ndarray):
                        max_x = x.max()
                    else:
                        max_x = max(x)
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
                if min_y is None:
                    if isinstance(y, np.ndarray):
                        min_y = y.min()
                    else:
                        min_y = min(y)
                if max_y is None:
                    if isinstance(y, np.ndarray):
                        max_y = y.max()
                    else:
                        max_y = max(y)
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
            tmp_labels_ticks_y = ax.get_yticklabels()#[1:-1:1]
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
                if min_y is None:
                    if isinstance(y, np.ndarray):
                        min_y = y.min()
                    else:
                        min_y = min(y)
                if max_y is None:
                    if isinstance(y, np.ndarray):
                        max_y = y.max()
                    else:
                        max_y = max(y)
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

    if legend:
        if isinstance(location_legend, np.ndarray):
            location_legend_form = location_legend.tolist()
        else:
            location_legend_form = location_legend

        ax.legend(loc=location_legend_form, fontsize=font_size_legend, ncols=n_columns_legend)

    if tight_layout:
        plt.tight_layout()


def single_figure_multi_plots_single_format(
        x, y, axes,
        widths_bars=0.6, bottoms_bars=None, aligns_bars='center', colors_bars=None,
        widths_bar_edges=None, colors_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        colors_error_bars=None, line_widths_error_bars=None,
        sizes_caps=0.0, thicknesses_caps=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_x=None, maxes_x=None, mins_y=None, maxes_y=None,
        legends=False, labels_legends=None, locations_legends='best', font_sizes_legends=None, n_columns_legends=1,
        id_figure=None, share_x='none', share_y='none',
        h_space=None, w_space=None,
        add_letters_to_titles=True, letter_start_titles=None, template_letter_addition_titles=None,
        n_pixels_x=300, n_pixels_y=300, logs=False, tight_layout=True):

    # todo: """ """
    """"""

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
    keys_axes_expected = np.asarray(['rows', 'columns', 'bars'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(axes, dict):
        raise TypeError('The type of "axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        axes, keys_axes_expected, values_axes_expected, name_dictionary='axes')
    cp_check.values_are_not_repeated(axes, name_dictionary='axes')

    axes_rc_in_data = np.asarray([axes['rows'], axes['columns']], dtype='i')
    axes_rc_in_data_sort = np.sort(axes_rc_in_data)
    n_axes_rc = axes_rc_in_data.size

    if isinstance(x, np.ndarray):
        x = x
    else:
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = np.asarray(y)

    shape_x = np.asarray(x.shape, dtype='i')
    n_dim_x = shape_x.size
    shape_y = np.asarray(y.shape, dtype='i')
    n_dim_y = shape_y.size

    indexes_rcp_i = np.empty(n_keys_axes_expected, dtype='O')

    if n_dim_y == n_keys_axes_expected:
        n_rows, n_columns = shape_y[axes_rc_in_data]
        shape_rc = shape_y[axes_rc_in_data_sort]

        if n_dim_x == n_keys_axes_expected:
            pass
        elif n_dim_x == 1:
            x = np.expand_dims(x, axis=axes_rc_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_y):
                    indexes_rcp_i[d] = slice(0, shape_y[d], 1)
                x_tmp = x
                x = np.empty(shape=shape_y, dtype=x.dtype)
                x[tuple(indexes_rcp_i)] = x_tmp
                shape_x = np.asarray(x.shape, dtype='i')
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))

    elif n_dim_y == 1:

        if n_dim_x == n_keys_axes_expected:
            n_rows, n_columns = shape_x[axes_rc_in_data]
            shape_rc = shape_x[axes_rc_in_data_sort]

            y = np.expand_dims(y, axis=axes_rc_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_x):
                    indexes_rcp_i[d] = slice(0, shape_x[d], 1)
                y_tmp = y
                y = np.empty(shape=shape_x, dtype=y.dtype)
                y[tuple(indexes_rcp_i)] = y_tmp
                shape_y = np.asarray(y.shape, dtype='i')

        elif n_dim_x == 1:
            n_rows = n_columns = 1
            shape_rc = np.asarray([1, 1], dtype='i')

            y = np.expand_dims(y, axis=axes_rc_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            x = np.expand_dims(x, axis=axes_rc_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))
    else:
        raise ValueError('y has to be a 1d or {}d array'.format(n_keys_axes_expected))

    n_subplots = n_rows * n_columns

    # format error_bars_x
    if error_bars_x is None:
        symmetric_error_bars_x = True
        indexes_error_bars_x = None

    elif not np.iterable(error_bars_x):
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
        error_bars_x = [error_bars_x] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_x, np.ndarray):
            error_bars_x = np.asarray(error_bars_x)

        shape_error_bars_x_tmp = np.asarray(error_bars_x.shape, dtype='i')
        n_dim_error_bars_x = shape_error_bars_x_tmp.size
        indexes_error_bars_x = np.empty(n_dim_error_bars_x, dtype='O')

        if n_dim_error_bars_x == n_keys_axes_expected:
            symmetric_error_bars_x = True
            shape_error_bars_x = shape_y

        elif n_dim_error_bars_x == (n_keys_axes_expected + 1):
            symmetric_error_bars_x = False
            shape_error_bars_x = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_x')

        for d in range(n_dim_error_bars_x):
            indexes_error_bars_x[d] = slice(0, shape_error_bars_x[d], 1)

        if not np.all(shape_error_bars_x_tmp == shape_error_bars_x):
            error_bars_x_tmp = error_bars_x
            error_bars_x = np.empty(shape=shape_error_bars_x, dtype=error_bars_x.dtype)
            error_bars_x[tuple(indexes_error_bars_x)] = error_bars_x_tmp

    # format error_bars_y
    if error_bars_y is None:
        symmetric_error_bars_y = True
        indexes_error_bars_y = None

    elif not np.iterable(error_bars_y):
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
        error_bars_y = [error_bars_y] * shape_y[axes['bars']]

    else:
        if not isinstance(error_bars_y, np.ndarray):
            error_bars_y = np.asarray(error_bars_y)

        shape_error_bars_y_tmp = np.asarray(error_bars_y.shape, dtype='i')
        n_dim_error_bars_y = shape_error_bars_y_tmp.size
        indexes_error_bars_y = np.empty(n_dim_error_bars_y, dtype='O')

        if n_dim_error_bars_y == n_keys_axes_expected:
            symmetric_error_bars_y = True
            shape_error_bars_y = shape_y

        elif n_dim_error_bars_y == (n_keys_axes_expected + 1):
            symmetric_error_bars_y = False
            shape_error_bars_y = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_y')

        for d in range(n_dim_error_bars_y):
            indexes_error_bars_y[d] = slice(0, shape_error_bars_y[d], 1)

        if not np.all(shape_error_bars_y_tmp == shape_error_bars_y):
            error_bars_y_tmp = error_bars_y
            error_bars_y = np.empty(shape=shape_error_bars_y, dtype=error_bars_y.dtype)
            error_bars_y[tuple(indexes_error_bars_y)] = error_bars_y_tmp

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
        widths_bars=widths_bars, bottoms_bars=bottoms_bars, aligns_bars=aligns_bars, colors_bars=colors_bars,
        colors_bar_edges=colors_bar_edges, widths_bar_edges=widths_bar_edges,
        colors_error_bars=colors_error_bars, line_widths_error_bars=line_widths_error_bars,
        sizes_caps=sizes_caps, thicknesses_caps=thicknesses_caps,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_x=mins_x, maxes_x=maxes_x, mins_y=mins_y, maxes_y=maxes_y,
        legends=legends, labels_legends=labels_legends, locations_legends=locations_legends,
        font_sizes_legends=font_sizes_legends, n_columns_legends=n_columns_legends, logs=logs)

    dict_parameters_rc = cp_format.format_shape_arguments(dict_parameters_rc, shape_rc)

    widths_bars = dict_parameters_rc['widths_bars']
    bottoms_bars = dict_parameters_rc['bottoms_bars']
    aligns_bars = dict_parameters_rc['aligns_bars']
    colors_bars = dict_parameters_rc['colors_bars']
    colors_bar_edges = dict_parameters_rc['colors_bar_edges']
    widths_bar_edges = dict_parameters_rc['widths_bar_edges']
    colors_error_bars = dict_parameters_rc['colors_error_bars']
    line_widths_error_bars = dict_parameters_rc['line_widths_error_bars']
    sizes_caps = dict_parameters_rc['sizes_caps']
    thicknesses_caps = dict_parameters_rc['thicknesses_caps']
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
    mins_x = dict_parameters_rc['mins_x']
    maxes_x = dict_parameters_rc['maxes_x']
    mins_y = dict_parameters_rc['mins_y']
    maxes_y = dict_parameters_rc['maxes_y']
    legends = dict_parameters_rc['legends']
    labels_legends = dict_parameters_rc['labels_legends']
    locations_legends = dict_parameters_rc['locations_legends']
    font_sizes_legends = dict_parameters_rc['font_sizes_legends']
    n_columns_legends = dict_parameters_rc['n_columns_legends']
    logs = dict_parameters_rc['logs']

    fig, ax = plt.subplots(
        n_rows, n_columns, sharex=share_x, sharey=share_y, squeeze=False,
        num=id_figure, frameon=False, dpi=my_dpi,
        figsize=((n_pixels_x * n_columns) / my_dpi, (n_pixels_y * n_rows) / my_dpi))

    indexes_rcp_i[axes['bars']] = slice(0, None, 1)

    indexes_rc_i = np.empty(n_axes_rc, dtype='O')

    if axes['rows'] < axes['columns']:
        indexes_combinations_rc = slice(0, n_axes_rc, 1)
    elif axes['rows'] > axes['columns']:
        indexes_combinations_rc = slice(-1, -(n_axes_rc + 1), -1)
    else:
        raise ValueError('axes')

    i = 0
    for combination_rc_i in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_rc):

        ax[tuple(combination_rc_i[indexes_combinations_rc])].tick_params(axis='both', labelbottom=True, labelleft=True)

        if indexes_error_bars_x is None:
            error_bars_x_i = error_bars_x
        else:
            if symmetric_error_bars_x:
                indexes_error_bars_x[axes_rc_in_data_sort] = combination_rc_i
            else:
                indexes_error_bars_x[axes_rc_in_data_sort + 1] = combination_rc_i
            error_bars_x_i = error_bars_x[tuple(indexes_error_bars_x)]

        if indexes_error_bars_y is None:
            error_bars_y_i = error_bars_y
        else:
            if symmetric_error_bars_y:
                indexes_error_bars_y[axes_rc_in_data_sort] = combination_rc_i
            else:
                indexes_error_bars_y[axes_rc_in_data_sort + 1] = combination_rc_i
            error_bars_y_i = error_bars_y[tuple(indexes_error_bars_y)]

        indexes_rcp_i[axes_rc_in_data_sort] = combination_rc_i
        tuple_indexes_rcp_i = tuple(indexes_rcp_i)

        indexes_rc_i[slice(0, n_axes_rc, 1)] = combination_rc_i
        tuple_indexes_rc_i = tuple(indexes_rc_i)
        if add_letters_to_titles:
            title_i = template_letter_addition_titles.format(
                subplot_letter=chr(num_letter_start_titles + i)) + titles[tuple_indexes_rc_i].tolist()
        else:
            title_i = titles[tuple_indexes_rc_i].tolist()

        single_figure_single_plot_single_format(
            x[tuple_indexes_rcp_i], y[tuple_indexes_rcp_i],
            width_bars=widths_bars[tuple_indexes_rc_i], bottom_bars=bottoms_bars[tuple_indexes_rc_i],
            align_bars=aligns_bars[tuple_indexes_rc_i], color_bars=colors_bars[tuple_indexes_rc_i],
            width_bar_edges=widths_bar_edges[tuple_indexes_rc_i], color_bar_edges=colors_bar_edges[tuple_indexes_rc_i],
            error_bars_x=error_bars_x_i, error_bars_y=error_bars_y_i,
            color_error_bars=colors_error_bars[tuple_indexes_rc_i],
            line_width_error_bars=line_widths_error_bars[tuple_indexes_rc_i],
            size_caps=sizes_caps[tuple_indexes_rc_i], thickness_caps=thicknesses_caps[tuple_indexes_rc_i],
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
            min_x=mins_x[tuple_indexes_rc_i], max_x=maxes_x[tuple_indexes_rc_i],
            min_y=mins_y[tuple_indexes_rc_i], max_y=maxes_y[tuple_indexes_rc_i],
            legend=legends[tuple_indexes_rc_i], label_legend=labels_legends[tuple_indexes_rc_i],
            location_legend=locations_legends[tuple_indexes_rc_i],
            font_size_legend=font_sizes_legends[tuple_indexes_rc_i],
            n_columns_legend=n_columns_legends[tuple_indexes_rc_i],
            ax=ax[tuple(combination_rc_i[indexes_combinations_rc])], log=logs[tuple_indexes_rc_i], tight_layout=False)

        i += 1

    if tight_layout:
        plt.tight_layout()

    if any([h_space is not None, w_space is not None]):
        plt.subplots_adjust(hspace=h_space, wspace=w_space)


def single_figure_multi_plots_multi_formats(
        x, y, axes,
        widths_all_formats=0.8, widths_between_formats=0.1,
        widths_bars=None, bottoms_bars=None, aligns_bars='center', colors_bars=None,
        widths_bar_edges=None, colors_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        colors_error_bars=None, line_widths_error_bars=None,
        sizes_caps=0.0, thicknesses_caps=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_x=None, maxes_x=None, mins_y=None, maxes_y=None,
        legends=False, labels_legends=None, locations_legends='best', font_sizes_legends=None, n_columns_legends=1,
        id_figure=None, share_x='none', share_y='none',
        h_space=None, w_space=None,
        add_letters_to_titles=True, letter_start_titles=None, template_letter_addition_titles=None,
        n_pixels_x=300, n_pixels_y=300, logs=False, tight_layout=True):

    # todo: """ """
    """"""

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
    keys_axes_expected = np.asarray(['rows', 'columns', 'formats', 'bars'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(axes, dict):
        raise TypeError('The type of "axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        axes, keys_axes_expected, values_axes_expected, name_dictionary='axes')
    cp_check.values_are_not_repeated(axes, name_dictionary='axes')

    axes_rc_in_data = np.asarray([axes['rows'], axes['columns']], dtype='i')
    axes_rc_in_data_sort = np.sort(axes_rc_in_data)
    n_axes_rc = axes_rc_in_data.size

    axes_rcj_in_data = np.asarray([axes['rows'], axes['columns'], axes['formats']], dtype='i')
    axes_rcj_in_data_sort = np.sort(axes_rcj_in_data)
    n_axes_rcj = axes_rcj_in_data.size

    if isinstance(x, np.ndarray):
        x = x
    else:
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = np.asarray(y)

    shape_x = np.asarray(x.shape, dtype='i')
    n_dim_x = shape_x.size
    shape_y = np.asarray(y.shape, dtype='i')
    n_dim_y = shape_y.size

    indexes_rcjp_i = np.empty(n_keys_axes_expected, dtype='O')

    if n_dim_y == n_keys_axes_expected:

        n_rows, n_columns, n_formats = shape_y[axes_rcj_in_data]
        shape_rc = shape_y[axes_rc_in_data_sort]
        shape_rcj = shape_y[axes_rcj_in_data_sort]

        if n_dim_x == n_keys_axes_expected:
            pass
        elif n_dim_x == 1:
            x = np.expand_dims(x, axis=axes_rcj_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_y):
                    indexes_rcjp_i[d] = slice(0, shape_y[d], 1)
                x_tmp = x
                x = np.empty(shape=shape_y, dtype=x.dtype)
                x[tuple(indexes_rcjp_i)] = x_tmp
                shape_x = np.asarray(x.shape, dtype='i')
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))

    elif n_dim_y == 1:

        if n_dim_x == n_keys_axes_expected:
            n_rows, n_columns, n_formats = shape_x[axes_rcj_in_data]
            shape_rc = shape_x[axes_rc_in_data_sort]
            shape_rcj = shape_x[axes_rcj_in_data_sort]

            y = np.expand_dims(y, axis=axes_rcj_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_x):
                    indexes_rcjp_i[d] = slice(0, shape_x[d], 1)
                y_tmp = y
                y = np.empty(shape=shape_x, dtype=y.dtype)
                y[tuple(indexes_rcjp_i)] = y_tmp
                shape_y = np.asarray(y.shape, dtype='i')

        elif n_dim_x == 1:
            n_rows = n_columns = n_formats = 1
            shape_rc = np.asarray([1, 1], dtype='i')
            shape_rcj = np.asarray([1, 1, 1], dtype='i')

            y = np.expand_dims(y, axis=axes_rcj_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            x = np.expand_dims(x, axis=axes_rcj_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))
    else:
        raise ValueError('y has to be a 1d or {}d array'.format(n_keys_axes_expected))

    n_subplots = n_rows * n_columns

    # format error_bars_x
    if error_bars_x is None:
        symmetric_error_bars_x = True
        indexes_error_bars_x = None

    elif not np.iterable(error_bars_x):
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
        error_bars_x = [error_bars_x] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_x, np.ndarray):
            error_bars_x = np.asarray(error_bars_x)

        shape_error_bars_x_tmp = np.asarray(error_bars_x.shape, dtype='i')
        n_dim_error_bars_x = shape_error_bars_x_tmp.size
        indexes_error_bars_x = np.empty(n_dim_error_bars_x, dtype='O')

        if n_dim_error_bars_x == n_keys_axes_expected:
            symmetric_error_bars_x = True
            shape_error_bars_x = shape_y

        elif n_dim_error_bars_x == (n_keys_axes_expected + 1):
            symmetric_error_bars_x = False
            shape_error_bars_x = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_x')

        for d in range(n_dim_error_bars_x):
            indexes_error_bars_x[d] = slice(0, shape_error_bars_x[d], 1)

        if not np.all(shape_error_bars_x_tmp == shape_error_bars_x):
            error_bars_x_tmp = error_bars_x
            error_bars_x = np.empty(shape=shape_error_bars_x, dtype=error_bars_x.dtype)
            error_bars_x[tuple(indexes_error_bars_x)] = error_bars_x_tmp

    # format error_bars_y
    if error_bars_y is None:
        symmetric_error_bars_y = True
        indexes_error_bars_y = None

    elif not np.iterable(error_bars_y):
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
        error_bars_y = [error_bars_y] * shape_y[axes['bars']]

    else:
        if not isinstance(error_bars_y, np.ndarray):
            error_bars_y = np.asarray(error_bars_y)

        shape_error_bars_y_tmp = np.asarray(error_bars_y.shape, dtype='i')
        n_dim_error_bars_y = shape_error_bars_y_tmp.size
        indexes_error_bars_y = np.empty(n_dim_error_bars_y, dtype='O')

        if n_dim_error_bars_y == n_keys_axes_expected:
            symmetric_error_bars_y = True
            shape_error_bars_y = shape_y

        elif n_dim_error_bars_y == (n_keys_axes_expected + 1):
            symmetric_error_bars_y = False
            shape_error_bars_y = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_y')

        for d in range(n_dim_error_bars_y):
            indexes_error_bars_y[d] = slice(0, shape_error_bars_y[d], 1)

        if not np.all(shape_error_bars_y_tmp == shape_error_bars_y):
            error_bars_y_tmp = error_bars_y
            error_bars_y = np.empty(shape=shape_error_bars_y, dtype=error_bars_y.dtype)
            error_bars_y[tuple(indexes_error_bars_y)] = error_bars_y_tmp

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

    dict_parameters_rcj = dict(
        widths_bars=widths_bars,  colors_bars=colors_bars,
        widths_bar_edges=widths_bar_edges, colors_bar_edges=colors_bar_edges,
        colors_error_bars=colors_error_bars, line_widths_error_bars=line_widths_error_bars,
        sizes_caps=sizes_caps, thicknesses_caps=thicknesses_caps,
        labels_legends=labels_legends)

    dict_parameters_rcj = cp_format.format_shape_arguments(dict_parameters_rcj, shape_rcj)

    widths_bars = dict_parameters_rcj['widths_bars']
    colors_bars = dict_parameters_rcj['colors_bars']
    widths_bar_edges = dict_parameters_rcj['widths_bar_edges']
    colors_bar_edges = dict_parameters_rcj['colors_bar_edges']
    colors_error_bars = dict_parameters_rcj['colors_error_bars']
    line_widths_error_bars = dict_parameters_rcj['line_widths_error_bars']
    sizes_caps = dict_parameters_rcj['sizes_caps']
    thicknesses_caps = dict_parameters_rcj['thicknesses_caps']
    labels_legends = dict_parameters_rcj['labels_legends']

    dict_parameters_rc = dict(
        widths_all_formats=widths_all_formats, widths_between_formats=widths_between_formats,
        bottoms_bars=bottoms_bars, aligns_bars=aligns_bars,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_x=mins_x, maxes_x=maxes_x, mins_y=mins_y, maxes_y=maxes_y,
        legends=legends, locations_legends=locations_legends, font_sizes_legends=font_sizes_legends,
        n_columns_legends=n_columns_legends, logs=logs)

    dict_parameters_rc = cp_format.format_shape_arguments(dict_parameters_rc, shape_rc)

    widths_all_formats = dict_parameters_rc['widths_all_formats']
    widths_between_formats = dict_parameters_rc['widths_between_formats']
    bottoms_bars = dict_parameters_rc['bottoms_bars']
    aligns_bars = dict_parameters_rc['aligns_bars']
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
    mins_x = dict_parameters_rc['mins_x']
    maxes_x = dict_parameters_rc['maxes_x']
    mins_y = dict_parameters_rc['mins_y']
    maxes_y = dict_parameters_rc['maxes_y']
    legends = dict_parameters_rc['legends']
    locations_legends = dict_parameters_rc['locations_legends']
    font_sizes_legends = dict_parameters_rc['font_sizes_legends']
    n_columns_legends = dict_parameters_rc['n_columns_legends']
    logs = dict_parameters_rc['logs']

    fig, ax = plt.subplots(
        n_rows, n_columns, sharex=share_x, sharey=share_y, squeeze=False,
        num=id_figure, frameon=False, dpi=my_dpi,
        figsize=((n_pixels_x * n_columns) / my_dpi, (n_pixels_y * n_rows) / my_dpi))

    indexes_rcjp_i[axes['bars']] = slice(0, None, 1)

    indexes_rc_i = np.empty(n_axes_rc, dtype='O')

    if axes['rows'] < axes['formats']:
        if axes['columns'] < axes['formats']:
            indexes_rc_in_indexes_rcj = np.asarray([0, 1], dtype='i')
        elif axes['columns'] > axes['formats']:
            indexes_rc_in_indexes_rcj = np.asarray([0, 2], dtype='i')
        else:
            raise ValueError('axes[\'columns\'] == axes[\'formats\']')
    elif axes['rows'] > axes['formats']:
        if axes['columns'] < axes['formats']:
            indexes_rc_in_indexes_rcj = np.asarray([0, 2], dtype='i')
        elif axes['columns'] > axes['formats']:
            indexes_rc_in_indexes_rcj = np.asarray([1, 2], dtype='i')
        else:
            raise ValueError('axes[\'columns\'] == axes[\'formats\']')
    else:
        raise ValueError('axes[\'rows\'] == axes[\'formats\']')

    indexes_rcj_i = np.asarray(
        [None if d in indexes_rc_in_indexes_rcj else slice(0, shape_rcj[d], 1)
         for d in range(0, n_axes_rcj, 1)],
        dtype='O')

    axes_next = {}
    if axes['formats'] < axes['bars']:
        axes_next['formats'] = 0
        axes_next['bars'] = 1
    elif axes['formats'] > axes['bars']:
        axes_next['formats'] = 1
        axes_next['bars'] = 0

    if axes['rows'] < axes['columns']:
        indexes_combinations_rc = slice(0, n_axes_rc, 1)
    elif axes['rows'] > axes['columns']:
        indexes_combinations_rc = slice(-1, -(n_axes_rc + 1), -1)
    else:
        raise ValueError('axes')

    i = 0
    for combination_rc_i in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_rc):

        ax[tuple(combination_rc_i[indexes_combinations_rc])].tick_params(axis='both', labelbottom=True, labelleft=True)

        if indexes_error_bars_x is None:
            error_bars_x_i = error_bars_x
        else:
            if symmetric_error_bars_x:
                indexes_error_bars_x[axes_rc_in_data_sort] = combination_rc_i
            else:
                indexes_error_bars_x[axes_rc_in_data_sort + 1] = combination_rc_i
            error_bars_x_i = error_bars_x[tuple(indexes_error_bars_x)]

        if indexes_error_bars_y is None:
            error_bars_y_i = error_bars_y
        else:
            if symmetric_error_bars_y:
                indexes_error_bars_y[axes_rc_in_data_sort] = combination_rc_i
            else:
                indexes_error_bars_y[axes_rc_in_data_sort + 1] = combination_rc_i
            error_bars_y_i = error_bars_y[tuple(indexes_error_bars_y)]

        indexes_rcjp_i[axes_rc_in_data_sort] = combination_rc_i
        tuple_indexes_rcjp_i = tuple(indexes_rcjp_i)

        indexes_rc_i[slice(0, n_axes_rc, 1)] = combination_rc_i
        tuple_indexes_rc_i = tuple(indexes_rc_i)
        if add_letters_to_titles:
            title_i = template_letter_addition_titles.format(
                subplot_letter=chr(num_letter_start_titles + i)) + titles[tuple_indexes_rc_i].tolist()
        else:
            title_i = titles[tuple_indexes_rc_i].tolist()

        indexes_rcj_i[indexes_rc_in_indexes_rcj] = combination_rc_i
        tuple_indexes_rcj_i = tuple(indexes_rcj_i)

        single_figure_single_plot_multi_formats(
            x[tuple_indexes_rcjp_i], y[tuple_indexes_rcjp_i], axes_next,
            width_all_formats=widths_all_formats[tuple_indexes_rc_i],
            width_between_formats=widths_between_formats[tuple_indexes_rc_i],
            bottom_bars=bottoms_bars[tuple_indexes_rc_i], align_bars=aligns_bars[tuple_indexes_rc_i],
            widths_bars=widths_bars[tuple_indexes_rcj_i],  colors_bars=colors_bars[tuple_indexes_rcj_i],
            widths_bar_edges=widths_bar_edges[tuple_indexes_rcj_i],
            colors_bar_edges=colors_bar_edges[tuple_indexes_rcj_i],
            error_bars_x=error_bars_x_i, error_bars_y=error_bars_y_i,
            colors_error_bars=colors_error_bars[tuple_indexes_rcj_i],
            line_widths_error_bars=line_widths_error_bars[tuple_indexes_rcj_i],
            sizes_caps=sizes_caps[tuple_indexes_rcj_i], thicknesses_caps=thicknesses_caps[tuple_indexes_rcj_i],
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
            min_x=mins_x[tuple_indexes_rc_i], max_x=maxes_x[tuple_indexes_rc_i],
            min_y=mins_y[tuple_indexes_rc_i], max_y=maxes_y[tuple_indexes_rc_i],
            legend=legends[tuple_indexes_rc_i], labels_legend=labels_legends[tuple_indexes_rcj_i],
            location_legend=locations_legends[tuple_indexes_rc_i],
            font_size_legend=font_sizes_legends[tuple_indexes_rc_i],
            n_columns_legend=n_columns_legends[tuple_indexes_rc_i],
            ax=ax[tuple(combination_rc_i[indexes_combinations_rc])], log=logs[tuple_indexes_rc_i], tight_layout=False)

        i += 1

    if tight_layout:
        plt.tight_layout()

    if any([h_space is not None, w_space is not None]):
        plt.subplots_adjust(hspace=h_space, wspace=w_space)


def multi_figures_single_plot_single_format(
        x, y, axes,
        widths_bars=0.6, bottoms_bars=None, aligns_bars='center', colors_bars=None,
        widths_bar_edges=None, colors_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        colors_error_bars=None, line_widths_error_bars=None,
        sizes_caps=0.0, thicknesses_caps=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_x=None, maxes_x=None, mins_y=None, maxes_y=None,
        legends=False, labels_legends=None, locations_legends='best', font_sizes_legends=None, n_columns_legends=1,
        id_figures=None, n_pixels_x=300, n_pixels_y=300, logs=False, tight_layouts=True):

    # todo: """ """
    """"""

    # format axes
    keys_axes_expected = np.asarray(['figures', 'bars'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(axes, dict):
        raise TypeError('The type of "axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        axes, keys_axes_expected, values_axes_expected, name_dictionary='axes')
    cp_check.values_are_not_repeated(axes, name_dictionary='axes')

    if isinstance(x, np.ndarray):
        x = x
    else:
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = np.asarray(y)

    shape_x = np.asarray(x.shape, dtype='i')
    n_dim_x = shape_x.size
    shape_y = np.asarray(y.shape, dtype='i')
    n_dim_y = shape_y.size

    indexes_fp_f = np.empty(n_keys_axes_expected, dtype='O')

    if n_dim_y == n_keys_axes_expected:
        n_figures = shape_y[axes['figures']]

        if n_dim_x == n_keys_axes_expected:
            pass
        elif n_dim_x == 1:
            x = np.expand_dims(x, axis=axes['figures'])
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_y):
                    indexes_fp_f[d] = slice(0, shape_y[d], 1)
                x_tmp = x
                x = np.empty(shape=shape_y, dtype=x.dtype)
                x[tuple(indexes_fp_f)] = x_tmp
                shape_x = np.asarray(x.shape, dtype='i')
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))

    elif n_dim_y == 1:

        if n_dim_x == n_keys_axes_expected:
            n_figures = shape_x[axes['figures']]
            y = np.expand_dims(y, axis=axes['figures'])
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_x):
                    indexes_fp_f[d] = slice(0, shape_x[d], 1)
                y_tmp = y
                y = np.empty(shape=shape_x, dtype=y.dtype)
                y[tuple(indexes_fp_f)] = y_tmp
                shape_y = np.asarray(y.shape, dtype='i')

        elif n_dim_x == 1:
            n_figures = 1
            y = np.expand_dims(y, axis=axes['figures'])
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            x = np.expand_dims(x, axis=axes['figures'])
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))
    else:
        raise ValueError('y has to be a 1d or {}d array'.format(n_keys_axes_expected))

    # format error_bars_x
    if error_bars_x is None:
        symmetric_error_bars_x = True
        indexes_error_bars_x = None

    elif not np.iterable(error_bars_x):
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
        error_bars_x = [error_bars_x] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_x, np.ndarray):
            error_bars_x = np.asarray(error_bars_x)

        shape_error_bars_x_tmp = np.asarray(error_bars_x.shape, dtype='i')
        n_dim_error_bars_x = shape_error_bars_x_tmp.size
        indexes_error_bars_x = np.empty(n_dim_error_bars_x, dtype='O')

        if n_dim_error_bars_x == n_keys_axes_expected:
            symmetric_error_bars_x = True
            shape_error_bars_x = shape_y

        elif n_dim_error_bars_x == (n_keys_axes_expected + 1):
            symmetric_error_bars_x = False
            shape_error_bars_x = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_x')

        for d in range(n_dim_error_bars_x):
            indexes_error_bars_x[d] = slice(0, shape_error_bars_x[d], 1)

        if not np.all(shape_error_bars_x_tmp == shape_error_bars_x):
            error_bars_x_tmp = error_bars_x
            error_bars_x = np.empty(shape=shape_error_bars_x, dtype=error_bars_x.dtype)
            error_bars_x[tuple(indexes_error_bars_x)] = error_bars_x_tmp

    # format error_bars_y
    if error_bars_y is None:
        symmetric_error_bars_y = True
        indexes_error_bars_y = None

    elif not np.iterable(error_bars_y):
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
        error_bars_y = [error_bars_y] * shape_y[axes['bars']]

    else:

        if not isinstance(error_bars_y, np.ndarray):
            error_bars_y = np.asarray(error_bars_y)

        shape_error_bars_y_tmp = np.asarray(error_bars_y.shape, dtype='i')
        n_dim_error_bars_y = shape_error_bars_y_tmp.size
        indexes_error_bars_y = np.empty(n_dim_error_bars_y, dtype='O')

        if n_dim_error_bars_y == n_keys_axes_expected:
            symmetric_error_bars_y = True
            shape_error_bars_y = shape_y

        elif n_dim_error_bars_y == (n_keys_axes_expected + 1):
            symmetric_error_bars_y = False
            shape_error_bars_y = np.append([2], shape_y, axis=0)

        else:
            raise ValueError('error_bars_y')

        for d in range(n_dim_error_bars_y):
            indexes_error_bars_y[d] = slice(0, shape_error_bars_y[d], 1)

        if not np.all(shape_error_bars_y_tmp == shape_error_bars_y):
            error_bars_y_tmp = error_bars_y
            error_bars_y = np.empty(shape=shape_error_bars_y, dtype=error_bars_y.dtype)
            error_bars_y[tuple(indexes_error_bars_y)] = error_bars_y_tmp

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

    dict_parameters_f = dict(
        widths_bars=widths_bars, bottoms_bars=bottoms_bars, aligns_bars=aligns_bars, colors_bars=colors_bars,
        widths_bar_edges=widths_bar_edges, colors_bar_edges=colors_bar_edges,
        colors_error_bars=colors_error_bars, line_widths_error_bars=line_widths_error_bars,
        sizes_caps=sizes_caps, thicknesses_caps=thicknesses_caps,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_x=mins_x, maxes_x=maxes_x, mins_y=mins_y, maxes_y=maxes_y,
        legends=legends, labels_legends=labels_legends, locations_legends=locations_legends,
        font_sizes_legends=font_sizes_legends, n_columns_legends=n_columns_legends,
        n_pixels_x=n_pixels_x, n_pixels_y=n_pixels_y, logs=logs, tight_layouts=tight_layouts)

    dict_parameters_f = cp_format.format_shape_arguments(dict_parameters_f, n_figures)

    widths_bars = dict_parameters_f['widths_bars']
    bottoms_bars = dict_parameters_f['bottoms_bars']
    aligns_bars = dict_parameters_f['aligns_bars']
    colors_bars = dict_parameters_f['colors_bars']
    widths_bar_edges = dict_parameters_f['widths_bar_edges']
    colors_bar_edges = dict_parameters_f['colors_bar_edges']
    colors_error_bars = dict_parameters_f['colors_error_bars']
    line_widths_error_bars = dict_parameters_f['line_widths_error_bars']
    sizes_caps = dict_parameters_f['sizes_caps']
    thicknesses_caps = dict_parameters_f['thicknesses_caps']
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
    mins_x = dict_parameters_f['mins_x']
    maxes_x = dict_parameters_f['maxes_x']
    mins_y = dict_parameters_f['mins_y']
    maxes_y = dict_parameters_f['maxes_y']
    legends = dict_parameters_f['legends']
    labels_legends = dict_parameters_f['labels_legends']
    locations_legends = dict_parameters_f['locations_legends']
    font_sizes_legends = dict_parameters_f['font_sizes_legends']
    n_columns_legends = dict_parameters_f['n_columns_legends']
    n_pixels_x = dict_parameters_f['n_pixels_x']
    n_pixels_y = dict_parameters_f['n_pixels_y']
    logs = dict_parameters_f['logs']
    tight_layouts = dict_parameters_f['tight_layouts']

    indexes_fp_f[axes['bars']] = slice(0, shape_y[axes['bars']], 1)

    for f in range(n_figures):

        fig_f = plt.figure(
            num=id_figures[f], frameon=False, dpi=my_dpi,
            figsize=(n_pixels_x[f] / my_dpi, n_pixels_y[f] / my_dpi))

        ax_f = plt.subplot(1, 1, 1)

        # ax_f.tick_params(axis='both', labelbottom=True, labelleft=True)

        indexes_fp_f[axes['figures']] = f

        if indexes_error_bars_x is None:
            error_bars_x_f = error_bars_x
        else:
            if symmetric_error_bars_x:
                indexes_error_bars_x[axes['figures']] = f
            else:
                indexes_error_bars_x[axes['figures'] + 1] = f
            error_bars_x_f = error_bars_x[tuple(indexes_error_bars_x)]

        if indexes_error_bars_y is None:
            error_bars_y_f = error_bars_y
        else:
            if symmetric_error_bars_y:
                indexes_error_bars_y[axes['figures']] = f
            else:
                indexes_error_bars_y[axes['figures'] + 1] = f
            error_bars_y_f = error_bars_y[tuple(indexes_error_bars_y)]

        single_figure_single_plot_single_format(
            x[tuple(indexes_fp_f)], y[tuple(indexes_fp_f)],
            width_bars=widths_bars[f], bottom_bars=bottoms_bars[f], align_bars=aligns_bars[f],
            color_bars=colors_bars[f],
            width_bar_edges=widths_bar_edges[f], color_bar_edges=colors_bar_edges[f],
            error_bars_x=error_bars_x_f, error_bars_y=error_bars_y_f,
            color_error_bars=colors_error_bars[f], line_width_error_bars=line_widths_error_bars[f],
            size_caps=sizes_caps[f], thickness_caps=thicknesses_caps[f],
            title=titles[f], font_size_title=font_sizes_titles[f], rotation_title=rotations_titles[f],
            label_x=labels_x[f], font_size_label_x=font_sizes_labels_x[f], rotation_label_x=rotations_labels_x[f],
            label_y=labels_y[f], font_size_label_y=font_sizes_labels_y[f], rotation_label_y=rotations_labels_y[f],
            labels_ticks_x=labels_ticks_x[f], ticks_x=ticks_x[f], n_ticks_x=n_ticks_x[f],
            stagger_labels_ticks_x=stagger_labels_ticks_x[f], font_size_labels_ticks_x=font_sizes_labels_ticks_x[f],
            rotation_labels_ticks_x=rotations_labels_ticks_x[f],
            labels_ticks_y=labels_ticks_y[f], ticks_y=ticks_y[f], n_ticks_y=n_ticks_y[f],
            stagger_labels_ticks_y=stagger_labels_ticks_y[f], font_size_labels_ticks_y=font_sizes_labels_ticks_y[f],
            rotation_labels_ticks_y=rotations_labels_ticks_y[f],
            min_x=mins_x[f], max_x=maxes_x[f], min_y=mins_y[f], max_y=maxes_y[f],
            legend=legends[f], label_legend=labels_legends[f], location_legend=locations_legends[f],
            font_size_legend=font_sizes_legends[f], n_columns_legend=n_columns_legends[f], ax=ax_f,
            n_pixels_x='not needed', n_pixels_y='not needed', log=logs[f], tight_layout=tight_layouts[f])


def multi_figures_multi_plots_single_format(
        x, y, axes,
        widths_bars=0.6, bottoms_bars=None, aligns_bars='center', colors_bars=None,
        widths_bar_edges=None, colors_bar_edges=None,
        error_bars_x=None, error_bars_y=None,
        colors_error_bars=None, line_widths_error_bars=None,
        sizes_caps=0.0, thicknesses_caps=None,
        titles=None, font_sizes_titles=None, rotations_titles=None,
        labels_x=None, font_sizes_labels_x=None, rotations_labels_x=None,
        labels_y=None, font_sizes_labels_y=None, rotations_labels_y=None,
        labels_ticks_x=None, ticks_x=None, n_ticks_x=None,
        stagger_labels_ticks_x=False, font_sizes_labels_ticks_x=None, rotations_labels_ticks_x=None,
        labels_ticks_y=None, ticks_y=None, n_ticks_y=None,
        stagger_labels_ticks_y=False, font_sizes_labels_ticks_y=None, rotations_labels_ticks_y=None,
        mins_x=None, maxes_x=None, mins_y=None, maxes_y=None,
        legends=False, labels_legends=None, locations_legends='best', font_sizes_legends=None, n_columns_legends=1,
        id_figures=None, share_x='none', share_y='none',
        h_spaces=None, w_spaces=None,
        add_letters_to_titles=True, letter_start_titles=None, template_letter_addition_titles=None,
        n_pixels_x=300, n_pixels_y=300, logs=False, tight_layouts=True):


    # format axes
    keys_axes_expected = np.asarray(['figures', 'rows', 'columns', 'bars'], dtype='U')
    n_keys_axes_expected = keys_axes_expected.size
    values_axes_expected = np.arange(start=0, stop=n_keys_axes_expected, step=1, dtype='i')
    if not isinstance(axes, dict):
        raise TypeError('The type of "axes" has to be dict')
    else:
        keys_axes, values_axes = cp_format.dict_to_key_array_and_value_array(axes)
        axes_negative = values_axes < 0
        values_axes[axes_negative] += n_keys_axes_expected
        for k in keys_axes[axes_negative]:
            axes[k] += n_keys_axes_expected

    cp_check.keys_necessary_known_and_values_necessary_known_exist_and_other_keys_and_values_do_not_exist(
        axes, keys_axes_expected, values_axes_expected, name_dictionary='axes')
    cp_check.values_are_not_repeated(axes, name_dictionary='axes')

    axes_frc_in_data = np.asarray([axes['figures'], axes['rows'], axes['columns']], dtype='i')
    axes_frc_in_data_sort = np.sort(axes_frc_in_data)
    n_axes_frc = axes_frc_in_data.size

    if isinstance(x, np.ndarray):
        x = x
    else:
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = np.asarray(y)

    shape_x = np.asarray(x.shape, dtype='i')
    n_dim_x = shape_x.size
    shape_y = np.asarray(y.shape, dtype='i')
    n_dim_y = shape_y.size

    indexes_frcp_i = np.empty(n_keys_axes_expected, dtype='O')

    if n_dim_y == n_keys_axes_expected:
        n_figures, n_rows, n_columns = shape_y[axes_frc_in_data]
        shape_frc = shape_y[axes_frc_in_data_sort]

        if n_dim_x == n_keys_axes_expected:
            pass
        elif n_dim_x == 1:
            x = np.expand_dims(x, axis=axes_frc_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_y):
                    indexes_frcp_i[d] = slice(0, shape_y[d], 1)
                x_tmp = x
                x = np.empty(shape=shape_y, dtype=x.dtype)
                x[tuple(indexes_frcp_i)] = x_tmp
                shape_x = np.asarray(x.shape, dtype='i')
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))

    elif n_dim_y == 1:

        if n_dim_x == n_keys_axes_expected:
            n_figures, n_rows, n_columns = shape_x[axes_frc_in_data]
            shape_frc = shape_x[axes_frc_in_data_sort]

            y = np.expand_dims(y, axis=axes_frc_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            if not np.all(shape_x == shape_y):
                for d in range(n_dim_x):
                    indexes_frcp_i[d] = slice(0, shape_x[d], 1)
                y_tmp = y
                y = np.empty(shape=shape_x, dtype=y.dtype)
                y[tuple(indexes_frcp_i)] = y_tmp
                shape_y = np.asarray(y.shape, dtype='i')

        elif n_dim_x == 1:
            n_figures = n_rows = n_columns = 1
            shape_frc = np.asarray([1, 1, 1], dtype='i')

            y = np.expand_dims(y, axis=axes_frc_in_data_sort.tolist())
            shape_y = np.asarray(y.shape, dtype='i')
            n_dim_y = shape_y.size
            x = np.expand_dims(x, axis=axes_frc_in_data_sort.tolist())
            shape_x = np.asarray(x.shape, dtype='i')
            n_dim_x = shape_x.size
        else:
            raise ValueError('x has to be a 1d or {}d array'.format(n_keys_axes_expected))
    else:
        raise ValueError('y has to be a 1d or {}d array'.format(n_keys_axes_expected))

    # format error_bars_x
    if error_bars_x is None:
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
    elif not np.iterable(error_bars_x):
        symmetric_error_bars_x = True
        indexes_error_bars_x = None
        error_bars_x = [error_bars_x] * shape_y[axes['bars']]
    else:
        if not isinstance(error_bars_x, np.ndarray):
            error_bars_x = np.asarray(error_bars_x)

        shape_error_bars_x_tmp = np.asarray(error_bars_x.shape, dtype='i')
        n_dim_error_bars_x = shape_error_bars_x_tmp.size
        indexes_error_bars_x = np.empty(n_dim_error_bars_x, dtype='O')

        if n_dim_error_bars_x == n_keys_axes_expected:
            symmetric_error_bars_x = True
            shape_error_bars_x = shape_y
        elif n_dim_error_bars_x == (n_keys_axes_expected + 1):
            symmetric_error_bars_x = False
            shape_error_bars_x = np.append([2], shape_y, axis=0)
        else:
            raise ValueError('error_bars_x')

        for d in range(n_dim_error_bars_x):
            indexes_error_bars_x[d] = slice(0, shape_error_bars_x[d], 1)

        if not np.all(shape_error_bars_x_tmp == shape_error_bars_x):
            error_bars_x_tmp = error_bars_x
            error_bars_x = np.empty(shape=shape_error_bars_x, dtype=error_bars_x.dtype)
            error_bars_x[tuple(indexes_error_bars_x)] = error_bars_x_tmp

    # format error_bars_y
    if error_bars_y is None:
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
    elif not np.iterable(error_bars_y):
        symmetric_error_bars_y = True
        indexes_error_bars_y = None
        error_bars_y = [error_bars_y] * shape_y[axes['bars']]
    else:
        if not isinstance(error_bars_y, np.ndarray):
            error_bars_y = np.asarray(error_bars_y)

        shape_error_bars_y_tmp = np.asarray(error_bars_y.shape, dtype='i')
        n_dim_error_bars_y = shape_error_bars_y_tmp.size
        indexes_error_bars_y = np.empty(n_dim_error_bars_y, dtype='O')

        if n_dim_error_bars_y == n_keys_axes_expected:
            symmetric_error_bars_y = True
            shape_error_bars_y = shape_y
        elif n_dim_error_bars_y == (n_keys_axes_expected + 1):
            symmetric_error_bars_y = False
            shape_error_bars_y = np.append([2], shape_y, axis=0)
        else:
            raise ValueError('error_bars_y')

        for d in range(n_dim_error_bars_y):
            indexes_error_bars_y[d] = slice(0, shape_error_bars_y[d], 1)

        if not np.all(shape_error_bars_y_tmp == shape_error_bars_y):
            error_bars_y_tmp = error_bars_y
            error_bars_y = np.empty(shape=shape_error_bars_y, dtype=error_bars_y.dtype)
            error_bars_y[tuple(indexes_error_bars_y)] = error_bars_y_tmp

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

    axes_next = {}
    for k in axes:
        if k == 'figures':
            continue
        if axes[k] < axes['figures']:
            axes_next[k] = axes[k]
        elif axes[k] > axes['figures']:
            axes_next[k] = axes[k] - 1
        else:
            raise ValueError('\n\tThe following condition is not met:\n'
                             '\t\taxes[\'{}\'] \u2260 axes[\'figures\']'.format(k))

    dict_parameters_frc = dict(
        widths_bars=widths_bars, bottoms_bars=bottoms_bars, aligns_bars=aligns_bars, colors_bars=colors_bars,
        colors_bar_edges=colors_bar_edges, widths_bar_edges=widths_bar_edges,
        colors_error_bars=colors_error_bars, line_widths_error_bars=line_widths_error_bars,
        sizes_caps=sizes_caps, thicknesses_caps=thicknesses_caps,
        titles=titles, font_sizes_titles=font_sizes_titles, rotations_titles=rotations_titles,
        labels_x=labels_x, font_sizes_labels_x=font_sizes_labels_x, rotations_labels_x=rotations_labels_x,
        labels_y=labels_y, font_sizes_labels_y=font_sizes_labels_y, rotations_labels_y=rotations_labels_y,
        labels_ticks_x=labels_ticks_x, ticks_x=ticks_x, n_ticks_x=n_ticks_x,
        stagger_labels_ticks_x=stagger_labels_ticks_x, font_sizes_labels_ticks_x=font_sizes_labels_ticks_x,
        rotations_labels_ticks_x=rotations_labels_ticks_x,
        labels_ticks_y=labels_ticks_y, ticks_y=ticks_y, n_ticks_y=n_ticks_y,
        stagger_labels_ticks_y=stagger_labels_ticks_y, font_sizes_labels_ticks_y=font_sizes_labels_ticks_y,
        rotations_labels_ticks_y=rotations_labels_ticks_y,
        mins_x=mins_x, maxes_x=maxes_x, mins_y=mins_y, maxes_y=maxes_y,
        legends=legends, labels_legends=labels_legends, locations_legends=locations_legends,
        font_sizes_legends=font_sizes_legends, n_columns_legends=n_columns_legends, logs=logs)

    dict_parameters_frc = cp_format.format_shape_arguments(dict_parameters_frc, shape_frc)

    widths_bars = dict_parameters_frc['widths_bars']
    bottoms_bars = dict_parameters_frc['bottoms_bars']
    aligns_bars = dict_parameters_frc['aligns_bars']
    colors_bars = dict_parameters_frc['colors_bars']
    colors_bar_edges = dict_parameters_frc['colors_bar_edges']
    widths_bar_edges = dict_parameters_frc['widths_bar_edges']
    colors_error_bars = dict_parameters_frc['colors_error_bars']
    line_widths_error_bars = dict_parameters_frc['line_widths_error_bars']
    sizes_caps = dict_parameters_frc['sizes_caps']
    thicknesses_caps = dict_parameters_frc['thicknesses_caps']
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
    mins_x = dict_parameters_frc['mins_x']
    maxes_x = dict_parameters_frc['maxes_x']
    mins_y = dict_parameters_frc['mins_y']
    maxes_y = dict_parameters_frc['maxes_y']
    legends = dict_parameters_frc['legends']
    labels_legends = dict_parameters_frc['labels_legends']
    locations_legends = dict_parameters_frc['locations_legends']
    font_sizes_legends = dict_parameters_frc['font_sizes_legends']
    n_columns_legends = dict_parameters_frc['n_columns_legends']
    logs = dict_parameters_frc['logs']

    dict_parameters_f = dict(
        share_x=share_x, share_y=share_y,
        h_spaces=h_spaces, w_spaces=w_spaces, add_letters_to_titles=add_letters_to_titles,
        n_pixels_x=n_pixels_x, n_pixels_y=n_pixels_y, tight_layouts=tight_layouts)

    dict_parameters_f = cp_format.format_shape_arguments(dict_parameters_f, [n_figures])

    share_x = dict_parameters_f['share_x']
    share_y = dict_parameters_f['share_y']
    h_spaces = dict_parameters_f['h_spaces']
    w_spaces = dict_parameters_f['w_spaces']
    add_letters_to_titles = dict_parameters_f['add_letters_to_titles']
    n_pixels_x = dict_parameters_f['n_pixels_x']
    n_pixels_y = dict_parameters_f['n_pixels_y']
    tight_layouts = dict_parameters_f['tight_layouts']

    indexes_frcp_f = np.asarray([slice(0, shape_y[d], 1) for d in range(0, n_keys_axes_expected, 1)], dtype='O')
    indexes_frc_f = np.asarray([slice(0, shape_frc[d], 1) for d in range(0, n_axes_frc, 1)], dtype='O')

    axis_f_in_parms_frc = 0
    if axes['figures'] > axes['rows']:
        axis_f_in_parms_frc += 1
    if axes['figures'] > axes['columns']:
        axis_f_in_parms_frc += 1

    for f in range(n_figures):

        if indexes_error_bars_x is None:
            error_bars_x_f = error_bars_x
        else:
            if symmetric_error_bars_x:
                indexes_error_bars_x[axes['figures']] = f
            else:
                indexes_error_bars_x[axes['figures'] + 1] = f
            error_bars_x_f = error_bars_x[tuple(indexes_error_bars_x)]

        if indexes_error_bars_y is None:
            error_bars_y_f = error_bars_y
        else:
            if symmetric_error_bars_y:
                indexes_error_bars_y[axes['figures']] = f
            else:
                indexes_error_bars_y[axes['figures'] + 1] = f
            error_bars_y_f = error_bars_y[tuple(indexes_error_bars_y)]

        indexes_frcp_f[axes['figures']] = f
        tuple_indexes_frcp_f = tuple(indexes_frcp_f)

        indexes_frc_f[axis_f_in_parms_frc] = f
        tuple_indexes_frc_f = tuple(indexes_frc_f)

        single_figure_multi_plots_single_format(
            x[tuple_indexes_frcp_f], y[tuple_indexes_frcp_f], axes=axes_next,
            widths_bars=widths_bars[tuple_indexes_frc_f], bottoms_bars=bottoms_bars[tuple_indexes_frc_f],
            aligns_bars=aligns_bars[tuple_indexes_frc_f], colors_bars=colors_bars[tuple_indexes_frc_f],
            widths_bar_edges=widths_bar_edges[tuple_indexes_frc_f],
            colors_bar_edges=colors_bar_edges[tuple_indexes_frc_f],
            error_bars_x=error_bars_x_f, error_bars_y=error_bars_y_f,
            colors_error_bars=colors_error_bars[tuple_indexes_frc_f],
            line_widths_error_bars=line_widths_error_bars[tuple_indexes_frc_f],
            sizes_caps=sizes_caps[tuple_indexes_frc_f], thicknesses_caps=thicknesses_caps[tuple_indexes_frc_f],
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
            mins_x=mins_x[tuple_indexes_frc_f], maxes_x=maxes_x[tuple_indexes_frc_f],
            mins_y=mins_y[tuple_indexes_frc_f], maxes_y=maxes_y[tuple_indexes_frc_f],
            legends=legends[tuple_indexes_frc_f], labels_legends=labels_legends[tuple_indexes_frc_f],
            locations_legends=locations_legends[tuple_indexes_frc_f],
            font_sizes_legends=font_sizes_legends[tuple_indexes_frc_f],
            n_columns_legends=n_columns_legends[tuple_indexes_frc_f],
            id_figure=id_figures[f], share_x=share_x[f], share_y=share_y[f],
            h_space=h_spaces[f], w_space=w_spaces[f],
            add_letters_to_titles=add_letters_to_titles[f], letter_start_titles=letter_start_titles,
            template_letter_addition_titles=template_letter_addition_titles,
            n_pixels_x=n_pixels_x[f], n_pixels_y=n_pixels_y[f], logs=logs[tuple_indexes_frc_f],
            tight_layout=tight_layouts[f])
