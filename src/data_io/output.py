import re
import numpy as np

import img
from context_class import Context, State
from scipy import signal
from data_io import plot
import json


def save_array(
        x: np.ndarray,
        var_name: str,
        context: Context,
        state: State,
        save_irls_stage=False,
        save_context=False,
        save_plot=False,
        verbose=False):
    """

    Parameters
    ----------
    x: np.ndarray
        Array to save
    context: Context
        Context object to use to provide appropriate meta_data
    state: State
        State object.
    var_name: str
        name of the variable
    save_context: bool
        optional. To determine whether the state and most of the context is
        saved as a .json file.

    Returns
    -------

    """
    res_dir = context.res_dir
    if verbose:
        con_metastr = context.data_cost_func + '_GradErr=' + str(
            context.grad_err) + \
                      '_alpha=' + str(context.current_reg_const) + '_'
        fname = res_dir + con_metastr + '_' + var_name + '_' + \
                str(state.total_warp_stages)
    else:
        con_metastr = ''
        fname = res_dir + var_name + '_' + str(state.total_warp_stages)

    if save_irls_stage:
        fname = fname + '_' + str(state.irls_stage)

    if context.gt_disparity is None:
        gt_height, gt_width, _ = context.gt_flows.shape
    else:
        gt_height, gt_width = context.gt_disparity.shape
    if len(x.shape) == 2:
        height = x.shape[0]
        width = x.shape[1]
    else:
        height = x.shape[2]
        width = x.shape[3]

    if var_name == 'Disparity':
        if (height, width) == (gt_height, gt_width):
            if context.save_weights:
                np.save(fname, x)
            else:
                np.save(fname, x.astype(np.half))
        else:
            x_scaled = signal.resample_poly(x, gt_height, height, axis=0,
                                            padtype='edge')
            x_scaled = signal.resample_poly(x_scaled, gt_width, width, axis=1,
                                            padtype='edge')
            if context.save_weights:
                np.save(fname, x_scaled)
            else:
                np.save(fname, x_scaled.astype(np.half))
    else:
        if context.save_weights:
            np.save(fname, x)
        else:
            np.save(fname, x.astype(np.half))

    if save_plot:
        if var_name == 'Disparity':
            plot_x = x  # * gt_height / height
        else:
            plot_x = x
        if verbose:
            con_metastr = context.data_cost_func + '_GradErr=' + str(
                context.grad_err) + \
                          '_alpha=' + str(context.current_reg_const) + '_'
            plot_name = res_dir + con_metastr + '_Plot_' + var_name + '_' + \
                        str(state.total_warp_stages)
        else:
            con_metastr = ''
            plot_name = res_dir + 'Plot_' + var_name + '_' + \
                        str(state.total_warp_stages)

        if save_irls_stage:
            plot_name = plot_name + '_' + str(state.irls_stage)

        plot.colorbar_img_plot(
            plot_x, title=var_name, save=True,
            filename=plot_name + '.png',
            # vmin=context.vmin,
            # vmax=context.vmax
            # vmin=0.0,
            # vmax=0.01
        )

    if save_context:
        con_name = res_dir + con_metastr + 'context' + '_' + \
                   str(state.total_warp_stages) + '.json'
        baselines = str(list(map(tuple, state.cur_baselines)))
        context_dictionary = {
            'Baselines': baselines,
            'Ground Truth Height': gt_height,
            'Ground Truth Width': gt_width,
            'Array Height': height,  # difference here indicates scaling
            'Arrary Width': width,

            'Min Expected Disparity': context.vmin,
            'Max Expected Disparity': context.vmax,

            'Data Term Cost Function': context.data_cost_func,
            'Quadratic Accumulation': context.quadratic_acc,
            'Regulariser Cost Function': context.reg_cost_func,
            'Regularisation Constant': context.current_reg_const,
            'Regularisation Sigma': context.reg_sigma,
            'Data Sigma': context.data_sigma,
            'Dynamic Data Sigma': context.data_sigma,
            'Normalised Opposite Weights': context.normalise_opposites,
            'Normalised Regulariser': context.reg_normalisation,
            'Four Neighbours': context.four_neighbours,

            'Gradient Error Technique used': context.grad_err,
            'Starting Number of Views': context.start_n,
            'Current Number of Views': state.current_n_views(context),

            'Max Coarse to Fine Levels': context.n_levels,
            'Current Coarse to Fine Level': state.ctf_level,
            'Total CTF Levels': state.total_ctf_levels,
            'Multi Res Levels': context.multi_res_levels,

            'Max Warps': str(context.n_warps_list),
            'Warp Mode': context.warp_mode,
            'Warp Top Level': context.warp_top_level,
            'Warping Boundary Condition': context.warp_boundary,
            'Current Warp': state.warp_stage,
            'Total Warps': state.total_warp_stages,
            'Interpolate Inside IRLS': context.interpolate_in_IRLS,
            'IRLS Stages': context.irls_stages,

            'Noise Added': context.add_noise,
            'Noise Sigma': context.noise_sigma,
            'Noise Seed Ref': context.noise_seed_ref,
            'Noise Seed Tgt': context.noise_seed_tgt,

            'Raw Images': context.raw_imgs,
            'Outside Image:': context.outside_image,
            'Median Filter': context.median_filter,
            'Gradient Scaling': context.grad_scaling,
            'View Scaling:': context.view_scaling,

            'Stability Check': context.stability_check,
        }

        with open(con_name, 'w') as con_file:
            json.dump(context_dictionary, con_file, indent=4)

        full_name = res_dir + con_metastr + 'context' + '_full_' + \
                    str(state.total_warp_stages) + '.json'
        with open(full_name, 'w') as full_file:
            full_dict = context.to_dict()
            json.dump(context.to_dict(), full_file, indent=4)


def save_list_arr(
        arrays: list,
        titles: list,
        context: Context,
        state: State,
        save_irls_list: list,
        save_context_list: list,
        save_plot_list: list,
        verbose_list: list
):
    """

    Parameters
    ----------
    arrays
    titles
    context
    state
    save_irls_list
    save_context_list
    save_plot_list
    verbose_list

    Returns
    -------

    """
    import itertools as it
    con_repeat = it.repeat(context)
    state_repeat = it.repeat(state)
    for args in zip(arrays, titles, con_repeat, state_repeat,
                    save_irls_list, save_context_list, save_plot_list,
                    verbose_list):
        x, title, context, state, save_irls, save_con, save_plot, verbose = args
        save_array(
            x, title, context, state, save_irls, save_con, save_plot,
            verbose
        )


def res_table(
        con_list: list,
        table_name: str
):
    """
    Produces a .csv table to summarise the results.

    Parameters
    ----------
    con_list
    table_name
    """
    import csv
    import os
    import re

    with open(table_name, 'w', newline='') as csvfile:
        results_writer = csv.writer(csvfile)
        results_writer.writerow([
            'Scene',
            'Method',
            'Reg_const',
            'Ax=b Solves',
            'RMSE',
            'Photometric Error'
        ])

        for context in con_list:
            gt = context.gt_disparity
            res_dir = context.res_dir
            scene, method, alpha = decode_res_path(res_dir)
            reg_const = context.current_reg_const
            if alpha != reg_const:
                raise ValueError('Alpha does not match with reg_const ')

            path, dirs, files = next(os.walk(res_dir))
            n_warps = context.total_max_warps
            total_solves = 0

            for warp in range(n_warps):
                n_irls_stages = 0
                match_str = 'Disparity_' + str(warp) + '_'
                for file in files:
                    if re.match(match_str, file):
                        n_irls_stages += 1
                for n in range(n_irls_stages):
                    f_name = res_dir + 'Disparity_' + str(warp) + '_' + \
                             str(n) + '.npy'
                    disparity = np.load(f_name)

                    err = disparity - gt
                    sq_err = np.abs(err) ** 2
                    rmse = np.sqrt(np.nanmean(sq_err))

                    photo_err = 0.0
                    # photo_err = photometric_err(disparity, context)
                    print(
                        scene, method, reg_const, total_solves, rmse,
                        photo_err
                    )

                    results_writer.writerow([
                        scene,
                        method,
                        str(reg_const),
                        str(total_solves),
                        str(rmse),
                        str(photo_err),
                    ])
                    total_solves += 1


def decode_res_path(path):
    """
    Takes inputs of the form:
        output/scratch/sideboard/grad_const/alpha=0.001/

    and returns the scene, the method and alpha

    Parameters
    ----------
    path

    Returns
    -------
    scene, meth, alpha

    """
    # \/[\w\d=.]+ is basically fwd slash and then a word, equals sign,
    # dec. point or digit, see regexr.com/72205
    dir_titles = re.findall('\/[\w\d=.-]+', path)

    # since '\alpha=blah' has b as its 7th character
    alpha_dir = dir_titles[-1]
    alpha_str = alpha_dir[7:]
    alpha = float(alpha_str)

    meth_dir = dir_titles[-2]
    # ignore first character
    meth = meth_dir[1:]

    scene_dir = dir_titles[-3]
    scene = scene_dir[1:]

    return scene, meth, alpha


def photometric_err(w: np.ndarray, context: Context, rms=True):
    """

    Parameters
    ----------
    w
    context

    Returns
    -------

    """
    from data_io import fileIO
    import multiprocessing as mp
    import shutil
    import os

    ref = fileIO.open_img(context.filenames[0])
    total_sq_err = np.zeros_like(ref)
    idx = 1

    input_list = []

    for name in context.filenames[1:]:
        tgt = fileIO.open_img(name)
        baseline = context.baselines[idx]
        i_args = ref, tgt, w, context.gt_disparity, context, baseline
        input_list.append(i_args)
        idx += 1

    err_list = map(pair_err, input_list)

    for err in err_list:
        total_sq_err += err ** 2

    total_sq_err /= idx
    err = np.sqrt(np.nanmean(total_sq_err))
    return err


def pair_err(args):
    ref, tgt_img, w, gt_disp, context, B = args
    ret = gt_based_pred_err(ref, tgt_img, w, gt_disp, context,
                            B, )
    return ret


def gt_based_pred_err(
        ref: np.ndarray,
        tgt: np.ndarray,
        wc: np.ndarray,
        true_disp: np.ndarray,
        parameters: Context,
        baseline: tuple,
):
    """

    Parameters
    ----------
    ref
    tgt
    wc
    true_disp
    parameters
    baseline
    direction
    """
    import warp

    b_x = -baseline[1]
    b_y = baseline[0]
    gt_disp = true_disp.astype(np.float64)
    occlusions = warp.reverse_map_wrapper(tgt, gt_disp, b_x, b_y)
    pred = warp.reverse_map_wrapper(tgt, wc, b_x, b_y)
    # print(pred)
    pred = np.where(np.isinf(occlusions), np.nan, pred)
    pred = np.where(np.isinf(pred), np.nan, pred)
    err = np.abs(ref - pred)
    return err


def pair_photometric_err(
        w: np.ndarray,
        ref: np.ndarray,
        tgt: np.ndarray,
        baseline: np.ndarray,
        context: Context):
    """
    Calculates the photometric error given a disparity and an image pair.
    In this case, we will use conventional warping.

    Parameters
    ----------
    context
    w: np.ndarray
        disparity
    ref: np.ndarray
        reference image
    tgt: np.ndarray
        target image
    baseline: np.ndarray
        the baseline vector between the reference and target image.

    Returns
    -------
    err: np.ndarray
    """
    import warp

    height, width = w.shape
    if w.shape != ref.shape or w.shape != tgt.shape:
        raise ValueError('Arrays are of incorrect shape')

    v = np.empty((height, width, 2))
    v[:, :, 0] = -baseline[1] * w
    v[:, :, 1] = baseline[0] * w

    warp_tgt = warp.fast_warp(tgt, v, context.warp_mode, context.warp_boundary)
    if context.outside_image:
        warp_tgt = img.outside_image_option(warp_tgt, ref, v)

    err = warp_tgt - ref
    return err
