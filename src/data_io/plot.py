# Module plot.py
from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import json
import ast
from context_class import Context


def colorbar_img_plot(img, title='', save=False, filename='', vmin=None,
                      vmax=None, cmap=None):
    """
    Plots an image with a colorbar.
    Parameters
    ----------
    img: np.ndarray
        image to plot
    title: str, optional
        the title of the plot.
    save: bool, optional
        If False (default), the image is not saved and is displayed. Otherwise,
        it is saved with the filename given by filename
    filename: str, optional
        Filename to save the plot as
    vmin: float
        the minimum value expected in the plot
    vmax: float
        the maximum value expected in the plot
    cmap: str
        the colourmap to use. See:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html

    """
    plt.figure()
    plt.title(title)
    ax = plt.gca()
    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def full_weights_plotter(weights_name: str, normalise=True, window=None,
                         view=None):
    """
    Plots all of the weights in one chart. Assumes that we have three scales and
    16 target views
    The views go like this (y, x)
    (1, 0)  (0, -1) (0, 1)  (-1, 0)
    (2, 0)  (0, -2) (0, 2)  (-2, 0)
    (3, 0)  (0, -3) (0, 3)  (-3, 0)
    (4, 0)  (0, -4) (0, 4)  (-4, 0)

    Parameters
    ----------
    weights_name
    normalise
    window: tuple or array of length 4
        (x_0, x_1, y_0, y_1)
    view: int
        which view we want to view

    Returns
    -------

    """
    # The baselines need to be labelled in (x, y) notation
    baselines_strs = [
        '(0, 1)', '( -1, 0)', '(1, 0)', '(0, -1)',
        '(0, 2)', '( -2, 0)', '(2, 0)', '(0, -2)',
        '(0, 3)', '( -3, 0)', '(3, 0)', '(0, -3)',
        '(0, 4)', '( -4, 0)', '(4, 0)', '(0, -4)'
    ]

    weights = np.abs(np.load(weights_name))

    n_dims = len(weights.shape)
    if n_dims != 4:
        raise ValueError('Weights has incompatible number of dimensions')

    n_imgs = weights.shape[0]
    n_res = weights.shape[1]
    if window is None:
        height = weights.shape[2]
        width = weights.shape[3]
    else:
        height = window[3] - window[2]
        width = window[1] - window[0]

    if n_imgs != 16 and n_imgs != 1:
        raise ValueError('Incompatible Number of Images')

    if normalise:
        norm_weights = np.empty_like(weights)
        total = np.sum(weights, axis=(0, 1)) / n_imgs
        for q in range(n_res):
            for p in range(n_imgs):
                norm_weights[p, q] = weights[p, q] / (total + 1e-10)
    else:
        norm_weights = weights

    if view is None:
        plot_weights = np.empty((height * 4, width * 4, n_res))
        for r in range(4):
            r_start = r * height
            r_end = (r + 1) * height
            for c in range(4):
                c_start = c * width
                c_end = (c + 1) * width
                img_idx = r * 4 + c
                for scale in range(n_res):
                    if window is None:
                        plot_weights[r_start:r_end, c_start:c_end, scale] = \
                            norm_weights[img_idx, scale, :, :]
                    else:
                        plot_weights[r_start:r_end, c_start:c_end, scale] = \
                            norm_weights[img_idx, scale, window[2]:window[3],
                            window[0]:window[1]]
    else:
        plot_weights = np.empty((height, width, n_res))
        for scale in range(n_res):
            plot_weights[:, :, scale] = norm_weights[view, scale,
                                        window[2]:window[3],
                                        window[0]:window[1]]

    figure, ax = plt.subplots()
    plt.imshow(plot_weights, vmin=0, vmax=2.0)
    plt.xticks([])
    plt.yticks([])

    offset_x = 20
    offset_y = 480

    str_loc = 0
    for r in range(4):
        for c in range(4):
            y_coord = r * (window[3] - window[2]) + offset_y
            x_coord = c * (window[1] - window[0]) + offset_x
            plt.text(x_coord, y_coord, baselines_strs[str_loc], color='w', size=25.0, fontstyle='normal')
            print(baselines_strs[str_loc])
            print(str_loc)
            str_loc += 1

    # plt.text(375, 50, '(-1, 0)')
    figure.set_size_inches(512 * 4 / 100, 512 * 4 / 100)
    plt.subplots_adjust(wspace=0., hspace=0., left=0., right=1,
                        bottom=0.0, top=1.)
    plt.show()


def reg_weights_plotter(weights_name: str, scale=1.0, window=None):
    """
    Plots the 01 regularisation weights. ie. each pixel with its upper neighbour

    Parameters
    ----------
    weights_name
    scale
    window

    Returns
    -------

    """
    weights = np.load(weights_name)

    n_dims = len(weights.shape)
    if n_dims != 4:
        raise ValueError('Weights has incompatible number of dimensions')

    if weights.shape[2] != 3 or weights.shape[3] != 3:
        raise ValueError('weights has incorrect graph dimensions')

    windowed = weights[window[2]:window[3], window[0]:window[1], 0, 1]
    scaled = np.abs(windowed) * scale
    colorbar_img_plot(scaled)


def full_reg_weights_plotter(weights_name: str, scale=1.0, window=None):
    """
    Plots the regularisation weights. These will look like
    00 01 02
    10 11 12
    20 21 22


    Parameters
    ----------
    weights_name
    window

    Returns
    -------

    """
    weights = np.load(weights_name)

    n_dims = len(weights.shape)
    if n_dims != 4:
        raise ValueError('Weights has incompatible number of dimensions')

    if window is None:
        height = weights.shape[0]
        width = weights.shape[1]
    else:
        height = window[3] - window[2]
        width = window[1] - window[0]

    if weights.shape[2] != 3 or weights.shape[3] != 3:
        raise ValueError('weights has incorrect graph dimensions')

    plot_weights = np.zeros((height * 3, width * 3))
    for r in range(3):
        r_start = r * height
        r_end = (r + 1) * height
        for c in range(3):
            c_start = c * width
            c_end = (c + 1) * width
            if r == 1 and c == 1:
                continue

            if window is None:
                plot_weights[r_start:r_end, c_start:c_end] = \
                    np.abs(weights[:, :, r, c]) * scale
            else:
                plot_weights[r_start:r_end, c_start:c_end] = \
                    np.abs(weights[window[2]:window[3], window[0]:window[1],
                           r, c]) * scale

    colorbar_img_plot(plot_weights)


def basic_weights_plotter(weights_name: str, context_name: str, plt_baselines,
                          normalise=False):
    """
    Plots the weights for a given set of baselines.

    Parameters
    ----------
    weights_name: str
        filename for the weights file.
    context_name: str
        filename for the context .json file.
    plt_baselines: list of tuples
        if there are weights for different levels of resolution then these
        tuples must be at least length 3 otherwise, it must be at least of
        length 2.
    normalise: bool, optional
        If true, we normalise the plot such that each pixel is normalised to add
        up to the number of images. (Default is False)

    Returns
    -------

    """
    weights = np.load(weights_name)
    with open(context_name) as f:
        con_data = json.load(f)

    if len(weights.shape) < 3:
        raise ValueError('Wrong number of dimensions in the weights file.')
    else:
        n_imgs = weights.shape[0]

    baselines = ast.literal_eval(con_data['Baselines'])
    n_dims = len(weights.shape)

    idx_bline_pairs = []
    for b in plt_baselines:
        if n_dims == 3 and len(b) < 2:
            raise ValueError('Baseline should have dimensions of least 2')
        if n_dims == 4 and len(b) < 3:
            raise ValueError('Baseline should have dimensions of at least 3')

        if (b[0], b[1]) in baselines:
            idx = baselines.index((b[0], b[1]))
            idx_bline_pairs.append((idx, b))
        idx = idx + 1

    print(baselines)
    print(idx_bline_pairs)

    if n_dims == 4:
        n_res = weights.shape[1]
    else:
        n_res = 1

    if normalise:
        norm_weights = np.empty_like(weights)
        scale = np.sum(weights, axis=(0, 1)) / (n_res * n_imgs)
        for q in range(n_res):
            for p in range(n_imgs):
                norm_weights[p, q] = weights[p, q] / (scale + 1e-4)
    else:
        norm_weights = weights

    for idx, baseline in idx_bline_pairs:
        if n_dims == 3:
            weights_view = norm_weights[idx - 1]
        elif n_dims == 4:
            weights_view = norm_weights[idx - 1, baseline[2]]
        else:
            raise ValueError('Wrong number of dimensions in the weights file.')

        vmax = np.amax(norm_weights)
        colorbar_img_plot(weights_view, title=str(baseline), vmin=0,
                          vmax=vmax,
                          cmap='binary')


def rmse_vs_matrix_solves_plot(context_list: list, constant=None, title='',
                               save=False, filename='', legend=None,
                               exclude_boundary=False, boundary=0, scale=None):
    """
    Plots the rmse value vs the number of matrix solves that have been done.

    Parameters
    ----------
    context_list
    title
    save
    filename
    legend
    exclude_boundary
    boundary

    Returns
    -------

    """
    n_ctxes = len(context_list)
    x_max = 0
    idx = 0
    for context in context_list:
        # Need to produce an rmse array for each of these.
        rmse_arr = rmse_vs_matrix_solves(context, exclude_boundary, boundary)
        tmp = np.count_nonzero(~np.isnan(rmse_arr))
        colors = list(mcolors.TABLEAU_COLORS)
        if tmp > x_max:
            x_max = tmp
        plt.plot(rmse_arr, color=colors[idx + 2])
        print(rmse_arr)
        idx += 1
    plt.title(title)

    if constant is not None:
        if type(constant) == float or type(constant) == int:
            plt.hlines(constant, xmin=0, xmax=x_max, colors=['Red'])
        else:
            idx = 0 + n_ctxes
            colors = list(mcolors.TABLEAU_COLORS)
            for const in constant:
                color = colors[idx]
                plt.hlines(const, xmin=0, xmax=x_max, colors=[color])
                idx += 1

    plt.xlabel('Ax = b Solves')
    if scale:
        plt.xscale(scale)

    plt.ylabel('RMSE')

    if legend is not None:
        plt.legend(legend)

    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def rmse_vs_warps_plot(context_list: list, title='', save=False, filename='',
                       legend=None, exclude_boundary=False, boundary=0):
    """
    Plots a set of different context's rmse values vs their warps.

    Parameters
    ----------
    context_list: list of context objects
        They must have the correct results directory and have
        matching parameters for the results which need to be opened.
    title: str
        Title to show in the plot
    save: bool
        whether to save the plot or not.
    filename: str
        filename to save as
    legend: list of strings
        strings to put in the legend
    Returns
    -------

    """

    for context in context_list:
        rmse_arr = rmse_vs_warp_vals(context, exclude_boundary=exclude_boundary,
                                     boundary=boundary)
        print(rmse_arr)
        plt.plot(rmse_arr)

    plt.title(title)

    plt.xlabel('Warps')
    plt.ylabel('RMSE')

    if legend is not None:
        plt.legend(legend)

    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def pred_err_plot(context_list: list, ref: np.ndarray,
                  tgt: np.ndarray, baseline: tuple | np.ndarray,
                  dir='fwd', title='',
                  save=False,
                  filename='',
                  legend=None):
    """

    Parameters
    ----------
    context_list
    ref
    tgt
    dir
    mode
    title
    save
    filename
    legend

    Returns
    -------

    """
    for context in context_list:
        pred_arr = pred_err_vs_warps(context, ref, tgt, baseline, dir)
        plt.plot(pred_arr)

    plt.title(title)
    plt.xlabel('Warps')
    plt.ylabel('Prediction Error')

    if legend is not None:
        plt.legend(legend)

    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def rmse_vs_warp_vals(context: Context, exclude_boundary=False, boundary=0):
    """
    For a given context open up all the .npy files containing the disparity
    and return their RMSE values in a numpy array

    the rest of the array

    Parameters
    ----------

    context: Context
        context object. Must have the correct results directory and have
        matching parameters for the results which need to be opened.
    exclude_boundary
    boundary

    Returns
    -------

    """
    max_warps = context.total_max_warps
    rmse_arr = np.full(max_warps, np.NaN)

    for n in range(0, max_warps):
        fname = context.res_dir + 'Disparity_' + str(n) + '.npy'
        try:
            disparity = np.load(fname)
            if exclude_boundary:
                # normal_err = disparity - context.gt_disparity
                b = boundary
                err = disparity[b:-b, b: -b] - context.gt_disparity[b:-b, b: -b]
            else:
                err = disparity - context.gt_disparity
            sq_err = np.abs(err) ** 2
            rmse_arr[n] = np.sqrt(np.nanmean(sq_err))
        except OSError:
            # Can't open the file.
            break

    return rmse_arr


def rmse_vs_matrix_solves(context: Context, exclude_boundary=False, boundary=0):
    """
    For a given context open up all the .npy files containing the disparity
    and return their RMSE values in a numpy array

    the rest of the array

    Parameters
    ----------

    context: Context
        context object. Must have the correct results directory and have
        matching parameters for the results which need to be opened.
    exclude_boundary
    boundary

    Returns
    -------

    """
    max_warps = context.total_max_warps
    max_solves = context.irls_stages
    rmse_arr = np.full(max_warps * max_solves, np.NaN)
    total_solves = 0

    for n in range(0, max_warps):
        fname = context.res_dir + 'Disparity_' + str(n)
        for m in range(0, max_solves):
            fname_mat_solve = fname + '_' + str(m) + '.npy'
            try:
                disparity = np.load(fname_mat_solve)
                if exclude_boundary:
                    # normal_err = disparity - context.gt_disparity
                    b = boundary
                    err = disparity[b:-b, b: -b] - context.gt_disparity[b:-b,
                                                   b: -b]
                else:
                    err = disparity - context.gt_disparity
                sq_err = np.abs(err) ** 2
                rmse_arr[total_solves] = np.sqrt(np.nanmean(sq_err))
            except OSError:
                # Can't open the file.
                break
            total_solves += 1

    return rmse_arr


def pred_err_vs_warps(context: Context, ref: np.ndarray, tgt: np.ndarray,
                      baseline, direction='fwd'):
    """
    Calculates prediction error vs warps for a given context.
    Parameters
    ----------
    context
    ref
    tgt
    baseline
    direction

    Returns
    -------

    """
    import simple
    fname_prefix = context.res_dir + 'Disparity_'
    max_warps = context.total_max_warps
    pred_arr = np.full(max_warps, np.NaN)

    for n in range(max_warps):
        fname = fname_prefix + str(n) + '.npy'
        try:
            disparity = np.load(fname)
            pred_arr[n] = simple.pred_err(disparity, baseline, ref, tgt,
                                          direction, 'rms')
        except OSError:
            break

    return pred_arr


def bad_px_vs_matrix_solve_plot(context_list: list, threshold: float,
                                title='', save=False,
                                filename='', legend=None, constant=None):
    n_ctxes = len(context_list)
    x_max = 0
    for context in context_list:
        # Need to produce an rmse array for each of these.
        bad_px_arr = bad_px_vs_matrix_solve(context, threshold) * 100
        tmp = np.count_nonzero(~np.isnan(bad_px_arr))
        if tmp > x_max:
            x_max = tmp
        plt.plot(bad_px_arr)
        print(bad_px_arr)

    if constant is not None:
        if type(constant) == float or type(constant) == int:
            plt.hlines(constant, xmin=0, xmax=x_max, colors=['Red'])
        else:
            idx = 0 + n_ctxes
            colors = list(mcolors.TABLEAU_COLORS)
            for const in constant:
                color = colors[idx]
                plt.hlines(const, xmin=0, xmax=x_max, colors=[color])
                idx += 1

    plt.title(title)

    plt.xlabel('Ax = b Solves')
    ylabel_str = '% Pixels with relative error >' + str(threshold) + '%'
    plt.ylabel(ylabel_str)

    if legend is not None:
        plt.legend(legend)
    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def bad_px_vs_matrix_solve(context: Context, threshold: float):
    """
    We use depth for this.
    Parameters
    ----------
    context
    threshold

    Returns
    -------

    """
    max_warps = context.total_max_warps
    max_solves = context.irls_stages
    bad_px = np.full(max_warps * max_solves, np.NaN)
    total_solves = 0

    for n in range(0, max_warps):
        fname = context.res_dir + 'Disparity_' + str(n)
        for m in range(0, max_solves):
            fname_mat_solve = fname + '_' + str(m) + '.npy'
            try:
                disparity = np.load(fname_mat_solve)
                # convert to depth
                num = context.dH * context.focal_length
                den = disparity + context.delta_x
                depth = num / den
                bad_px[total_solves] = bad_px_percent(threshold, depth,
                                                      context.gt_depth)
            except OSError:
                # Can't open the file.
                break
            total_solves += 1

    return bad_px


def bad_px_percent(threshold: float, depth: np.ndarray,
                   gt_depth: np.ndarray):
    """
    Calculates the percentage of pixels that have an error greater than
    threshold %.

    Parameters
    ----------
    threshold: float
        the proportional error in %
    depth: np.ndarray
        disparity array
    gt_depth: np.ndarray
        ground-truth disparity

    Returns
    -------
    float
        the percentage of pixels that have an error greater than
        threshold %.
    """
    diff = np.abs(depth - gt_depth)
    ratio_pc = 100 * diff / np.abs(gt_depth)  # this isn't right.
    res_arr = np.where(ratio_pc > threshold, 1.0, 0.0)
    return np.sum(res_arr) / np.size(depth)


def plot_img_tight(img, pos, vmin=None, vmax=None, cbar=False, ylabel=None,
                   xlabel=None, ticks=False, cmap='viridis'):
    """
    Plots images as subplots in a tight layout.

    Parameters
    ----------
    xlabel
    img
    pos
    vmin
    vmax
    cbar
    ylabel
    ticks

    Returns
    -------

    """
    ax1 = plt.subplot(pos[0], pos[1], pos[2])
    ax1.set_aspect('equal')
    # plt.axis('off')
    im_plot = plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

    if cbar is True:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("top", size="5%", pad=0.25)
        plt.colorbar(mappable=im_plot, cax=cax, orientation='horizontal')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
