# Module multiview.py

"""
Handles the main loop which deals with the disparity calculation using multiview
data.
"""
from __future__ import annotations

import itertools

import numpy as np
from img import Images, ResampledImgs, RawImages, resample_2d, \
    TopScaleSplines, GaussianSplines, OverSampledImgs
from context_class import Context, State
import cgrad as cg
from graph import FullWeightedGraph
from data_io import output as out
from scipy import signal, ndimage
import weights
import cost
import copy
import simple


def mv_disparity(context: Context, state=None, w=None):
    """
    Calculates the disparity according to the parameters in context.

    Parameters
    ----------
    context: Context

    Returns
    -------

    """

    n_levels = context.n_levels
    n_res = context.multi_res_levels
    n_scales = n_res * n_levels
    alpha = context.current_reg_const

    data_cost_func = context.data_cost_func
    reg_cost_func = context.reg_cost_func

    # if data_cost_func == 'L2' and reg_cost_func == 'L2':
    #     irls_stages = 1
    # else:
    irls_stages = context.irls_stages
    tol = context.tol

    max_iter = context.max_iter
    if state is None:
        state = State(0, n_scales)
    state.vmin = context.vmin
    state.cur_baselines = context.baselines
    state.vmax = context.vmax
    height = context.gt_disparity.shape[0]
    width = context.gt_disparity.shape[1]
    reg_sigma = context.reg_sigma
    if w is None and context.w_init is None:
        w = np.zeros((height, width))
    elif w is None:
        w = resample_2d(context.w_init, height, width)
    else:
        w = np.asarray(w, dtype=np.float64)
        w = resample_2d(w, height, width)

    ctf_levels = n_scales - n_res + 1

    for ctf_level in range(ctf_levels):
        # level = n_levels - ctf_level - 1
        level = n_scales - ctf_level - n_res
        print('Level: ', level)
        scaled_alpha = alpha * (2 ** level)
        print(scaled_alpha)
        # state.start = level * n_res
        state.start = level
        state.stop = level + n_res

        images = RawImages(context, state)

        img_height = images.height
        img_width = images.width
        max_q = images.n_res
        n_targets = images.n_targets

        dot_grads = np.empty((n_targets, max_q, img_height, img_width))
        raw_grads = np.empty((n_targets + 1, max_q, img_height, img_width, 2))
        diff = np.empty_like(dot_grads)

        if w.shape[0] != img_height:
            w = resample_2d(w, img_height, img_width)
            print('Resampled w')

        for q in range(max_q):
            raw_grads[0, q] = images.get_grad(0, q, w, level)

        print('CTF Stage:', state.ctf_level)
        n_warps = context.n_warps_list[ctf_level]
        n = 0
        continue_irls = True
        state.update_limit()
        state.update_ctf_level(ctf_level)
        print('Limit: ', state.disp_lim)

        while n < n_warps and continue_irls and \
                state.ctf_level < context.total_max_warps:
            print('Warp Stage:', state.total_warp_stages)
            state.update_warp_stage(n)

            if not context.interpolate_in_IRLS:
                dw = np.zeros_like(w)
                prev_dw = np.copy(dw)
                update_data(images, w, level, raw_grads, dot_grads, diff)
                scale = update_scale(context, raw_grads, n_targets)
                W_pq, err, grad_err, coarse_err, delta_w = \
                    update_d_weights(
                        context, state, images, w, dot_grads, diff, raw_grads
                    )

                if context.confidence_metric:
                    confidence = weights.calc_confidence(
                        grad_err, coarse_err, dot_grads, context.baselines
                    )
                else:
                    confidence = 1.0

                if context.dynamic_d_sig:
                    state.calculate_sig_d(dot_grads, diff, w)
                else:
                    state.current_d_sig = context.data_sigma
                if context.save_weights:
                    out.save_array(
                        W_pq, 'Weights', context, state,
                        save_irls_stage=True
                    )

            for i in range(irls_stages):
                print('Ax = b Solves:', i)
                state.update_irls_stage(i)
                if context.interpolate_in_IRLS:
                    dw = np.zeros_like(w)
                    prev_dw = np.copy(dw)
                    update_data(images, w, level, raw_grads, dot_grads, diff)
                    scale = update_scale(context, raw_grads, n_targets)
                    W_pq, err, grad_err, coarse_err, delta_w = \
                        update_d_weights(
                            context, state, images, w, dot_grads, diff,
                            raw_grads
                        )
                    if context.confidence_metric:
                        confidence = weights.calc_confidence(
                            grad_err, coarse_err, dot_grads, context.baselines
                        )
                    else:
                        confidence = 1.0

                    if context.dynamic_d_sig and context.data_cost_func == 'Welsch':
                        state.calculate_sig_d(dot_grads, diff, w)
                    else:
                        state.current_d_sig = context.data_sigma

                if data_cost_func == 'L2' or state.is_first_stage():
                    irls_weights = np.ones_like(W_pq)
                else:
                    irls_weights = weights.robust_weights(
                        dot_grads, diff, dw, scale, context, state
                    )
                total_weights = irls_weights * W_pq * scale
                sum_g_pq = np.sum(
                    dot_grads ** 2 * total_weights * confidence, axis=(0, 1)
                )
                sum_delta_Ipq = np.sum(
                    -diff * dot_grads * total_weights * confidence, axis=(0, 1)
                )

                reg = FullWeightedGraph(img_height, img_width, sig_r=reg_sigma)

                if reg_cost_func != 'L2' and not state.is_first_stage():
                    if reg_cost_func == 'L1' and context.four_neighbours:
                        reg.set_weights_4_neighbours_l1(w + dw, level)
                    else:
                        reg.set_weights(w + dw, reg_sigma, reg_cost_func)

                if context.reg_normalisation:
                    reg.normalise(W_pq)

                reg.apply_w_kernel(scaled_alpha, context.four_neighbours)
                print('sigma_d = ', state.current_d_sig)
                dw = cg.cg_disparity(
                    w, dw, sum_g_pq, sum_delta_Ipq, reg, max_iter, tol
                )
                if context.stability_check and context.interpolate_in_IRLS:
                    ret = cost_stability(
                        total_weights, images, dot_grads, diff, w, dw,
                        state.disp_lim, alpha, context, level=level
                    )
                    dw, state.disp_lim, continue_irls = ret

                if np.amax(np.abs(prev_dw - dw)) < tol:
                    # If there hasn't been any significant change in the
                    # disparity field then no more irls stages are needed
                    continue_irls = False
                prev_dw = np.copy(dw)

                if context.save_weights:
                    w_cost_d, w_cost_r = cost.point_wise_cost(
                        W_pq, irls_weights, dot_grads, diff, w,
                        scaled_alpha
                    )
                    print('Data Cost w:', np.sum(w_cost_d))

                    dw_cost_d, dw_cost_r = cost.point_wise_cost(
                        W_pq, irls_weights, dot_grads, diff, dw,
                        scaled_alpha
                    )
                    print('Data Cost dw:', np.sum(dw_cost_d))

                    cost_diff_d = w_cost_d - dw_cost_d
                    cost_diff_d = np.where(cost_diff_d > 0, np.nan, cost_diff_d)

                    total_cost_w = w_cost_d + w_cost_r

                    total_cost_dw = dw_cost_r + dw_cost_d

                    total_cost_diff = total_cost_w - total_cost_dw
                    total_cost_diff = np.where(
                        total_cost_diff > 0, np.nan, total_cost_diff
                    )

                    print('Diff:', np.sum(cost_diff_d))
                    data_term_vpq = sum_delta_Ipq / sum_g_pq

                    arrays = [
                        w_cost_d,
                        w_cost_r,
                        cost_diff_d,
                        total_cost_w,
                        total_cost_dw,
                        total_cost_diff,
                        dw_cost_d,
                        dw_cost_r,
                        W_pq,
                        dot_grads ** 2 * total_weights,
                        # dot_grads ** 2,
                        ndimage.gaussian_filter(dot_grads, simple.get_sigma(0)) ** 2,
                        -diff * dot_grads * total_weights,
                        # -diff * dot_grads,
                        -ndimage.gaussian_filter(diff, simple.get_sigma(0)) *
                        ndimage.gaussian_filter(dot_grads, simple.get_sigma(0)),
                        sum_g_pq,
                        sum_delta_Ipq,
                        data_term_vpq,
                        reg.graph,
                        err,
                        grad_err,
                        coarse_err,
                        delta_w,
                        np.sum(W_pq, axis=(0, 1))
                    ]
                    titles = [
                        'Data_Cost_w',
                        'Reg_Cost_w',
                        'Data_Cost_diff',
                        'Total_Cost_w',
                        'Total_Cost_dw',
                        'Total_Cost_diff',
                        'Data_Cost_dw',
                        'Reg_Cost_dw',
                        'Weights',
                        'Weighted_Grads',
                        'Grads_Unweighted',
                        'Weighted_Diffs',
                        'Diffs_Unweighted',
                        'sum_gpq',
                        'sum_Delta_Ipq',
                        'v_pq',
                        'reg_graph',
                        'Error',
                        'Grad_Error',
                        'CoarseError',
                        'Delta_w',
                        'sum_Wpq',
                    ]
                    save_irls_list = itertools.repeat(True)
                    save_context_list = itertools.repeat(False)
                    save_plot_list = []
                    no_plot_set = {
                        'Weights', 'Weighted_Grads',
                        'Grads_Unweighted', 'Weighted_Diffs',
                        'Diffs_Unweighted', 'reg_graph', 'Error',
                        'Grad_Error', 'CoarseError', 'Delta_w'
                    }
                    for title in titles:
                        if title in no_plot_set:
                            save_plot_list.append(False)
                        else:
                            save_plot_list.append(True)

                    verbose_list = itertools.repeat(False)

                    out.save_list_arr(
                        arrays, titles, context, state, save_irls_list,
                        save_context_list, save_plot_list, verbose_list
                    )

                    images.save_warped_imgs(context.res_dir, w,
                                            state.irls_stage, level)

                if context.interpolate_in_IRLS:
                    if context.apply_limit:
                        dw, is_limited = limit_disparity(dw, state, context)
                        if ctf_level == n_levels - 1:
                            continue_irls = True
                        else:
                            continue_irls = is_limited

                    w += dw
                    if context.median_filter:
                        w = ndimage.median_filter(w, size=5)

                    out.save_array(
                        w, 'Disparity', context, state,
                        save_context=True, save_plot=True,
                        save_irls_stage=True
                    )

                else:
                    out.save_array(
                        w + dw, 'Disparity', context, state,
                        save_context=True, save_plot=True,
                        save_irls_stage=True
                    )

                if continue_irls is False:
                    break

            if not context.interpolate_in_IRLS:
                if context.stability_check:
                    ret = cost_stability(
                        total_weights, images, dot_grads, diff, w, dw,
                        state.disp_lim, alpha, context, level=level
                    )
                    dw, state.disp_lim, continue_irls = ret
                if context.apply_limit:
                    dw, is_limited = limit_disparity(dw, state, context)
                    if ctf_level == n_levels - 1:
                        continue_irls = True
                    else:
                        continue_irls = is_limited
                w = dw + w
                if context.median_filter:
                    w = ndimage.median_filter(w, size=5)
            n += 1

    return w, state


def update_d_weights(
        context: Context,
        state: State,
        images: Images,
        w: np.ndarray,
        dot_grads: np.ndarray,
        diff: np.ndarray,
        raw_grads: np.ndarray
):
    """
    Updates the data term weights.

    Parameters
    ----------
    context
    state
    images
    w
    dot_grads
    diff
    raw_grads

    Returns
    -------

    """
    n_targets = images.n_targets
    max_q = images.n_res
    height = images.height
    width = images.width

    if context.grad_err:
        if context.delta_w_vpq:
            ret_val = weights.stable_gc_weights(
                images, raw_grads, dot_grads, diff, w, state, context
            )
            d_weights, err, grad_err, coarse_err, delta_w = ret_val
        else:
            raise(ValueError('Delta w must be data determined.'))

        # if not context.reg_normalisation:
        d_weights = weights.normalise_weights(
            dot_grads, d_weights, images, w, context
        )
    else:
        d_weights = np.zeros((n_targets, max_q, height, width))
        err = np.zeros_like(d_weights)
        grad_err = np.zeros_like(d_weights)
        coarse_err = np.zeros_like(d_weights)
        delta_w = np.zeros_like(d_weights)
        q = state.stop - state.start - 1
        d_weights[:, q] = 1.0
    return d_weights, err, grad_err, coarse_err, delta_w


def update_data(images: Images, w: np.ndarray, level: int,
                raw_grads: np.ndarray, dot_grads: np.ndarray, diff: np.ndarray,
                ):
    """
    This updates the raw_grads, dot_grads and diff variables from the image data

    Parameters
    ----------
    images: Images
        Images object which contains the raw or spline coefficient data to
        interpolated.
    w: np.ndarray
        the disparity field
    level: int
        which resolution,
    raw_grads: np.ndarray
        the output array where the raw gradients go
    dot_grads: np.ndarray
        the output array where the g_pq values go
    diff: np.ndarray
        the output array where delta_Ipq goes.

    Returns
    -------
    None
    """
    n_targets = images.n_targets
    max_q = images.n_res
    for p in range(1, n_targets + 1):
        for q in range(max_q):
            raw_grads[p, q] = images.get_grad(p, q, w, level)
            dot_grads[p - 1, q] = images.get_dot_grad(
                p, q, w, level
            )
            diff[p - 1, q] = images.get_diff(p, q, w, level)
    return


def update_scale(context: Context, raw_grads: np.ndarray, n_targets: int):
    """
    Calculates the scale which is applied to the data.

    Parameters
    ----------
    context
    raw_grads
    n_targets

    Returns
    -------

    """
    n_res = raw_grads.shape[1]
    height = raw_grads.shape[2]
    width = raw_grads.shape[3]
    if context.grad_scaling:
        mag_grads = np.empty((n_targets, n_res, height, width))
        update_mag_grads(raw_grads, n_targets, mag_grads)
        g_scale = np.reciprocal(mag_grads)
    else:
        g_scale = 1.0
    if context.view_scaling:
        v_scale = 1.0 / n_targets
    else:
        v_scale = 1.0

    scale = g_scale * v_scale
    return scale


def update_mag_grads(
        raw_grads: np.ndarray,
        n_targets: int,
        mag_grads: np.ndarray
):
    """
    Calculates the magnitdue of the gradients from the raw gradient data

    Parameters
    ----------
    raw_grads: np.ndarray
        the gradients used to calculate the magnitude of the gradients
    n_targets: np.ndarray
        the number of target images/views
    mag_grads: np.ndarray
        the output array where the magnitude of gradients is placed

    Returns
    -------

    """
    for p in range(n_targets):
        tmp = (raw_grads[0] + raw_grads[p + 1]) / 2.0
        tmp_0 = tmp[:, :, :, 0]
        tmp_1 = tmp[:, :, :, 1]
        mag_grads[p] = tmp_0 ** 2 + tmp_1 ** 2 + 0.0001


def limit_disparity(disparity: np.ndarray, state: State, context: Context):
    """
    Tells you whether the disparity has been limited and limits the disparity
    Parameters
    ----------
    disparity
    state
    context

    Returns
    -------
    limited_disparity, is_limited
    """
    n_levels = context.n_levels
    ctf_level = state.ctf_level
    lim = state.disp_lim
    if np.amax(np.abs(disparity)) > lim:
        limited_disp = np.where(disparity > lim, lim, disparity)
        limited_disp = np.where(limited_disp < -lim, -lim, limited_disp)
        return limited_disp, True
    else:
        return disparity, False


def cost_stability(
        Wpq: np.ndarray,
        images: Images | ResampledImgs,
        dot_grad: np.ndarray, diff: np.ndarray,
        w: np.ndarray,
        dw: np.ndarray,
        limit: int | float,
        alpha: float,
        context: Context,
        level=0
):
    """
    Limits the last step and then calculates whether the cost function has gone
    down or not as a result of the last step. If the last step has not reduced
    the cost, we reduce the limit.

    We determine the cost function as per the paper, but we use the weights
    to implement the robust cost function.

    Parameters
    ----------
    Wpq
    dot_grad
    diff
    dw
    alpha
    limit
    context
    level

    Returns
    -------

    """
    zero_arr = np.zeros_like(dw)
    e_d_prev = cost.data_cost(Wpq, dot_grad, diff, zero_arr)
    # e_d_prev = out.photometric_err(w, context)
    e_r_prev = cost.reg_cost(w) * alpha
    prev = e_d_prev + e_r_prev
    print('Cost of w:', e_d_prev, '+', e_r_prev, '=', prev)
    cur_lim = limit
    cur = np.inf

    new_diff = np.empty_like(diff)
    new_grad = np.empty_like(dot_grad)

    while cur > prev:
        lim = cur_lim * local_limit(Wpq, dot_grad, images)
        lim_dw = np.where(dw > lim, lim, dw)
        lim_dw = np.where(lim_dw < -lim, -lim, lim_dw)
        # lim_diff = np.abs(dw - lim_dw)
        # print(np.amax(lim_diff))
        hyp_w = ndimage.median_filter(w + lim_dw, size=5)
        for p in range(images.n_targets):
            for q in range(images.n_res):
                new_grad[p, q] = images.get_dot_grad(
                    p + 1, q, hyp_w, level
                )
                new_diff[p, q] = images.get_diff(p + 1, q, hyp_w, level)

        e_d_cur = cost.data_cost(Wpq, new_grad, new_diff, zero_arr)
        # e_d_cur = out.photometric_err(hyp_w, context)
        e_r_cur = cost.reg_cost(hyp_w) * alpha
        cur = e_d_cur + e_r_cur
        print('Cost of w + dw:', e_d_cur, '+', e_r_cur, '=', cur)
        if cur > prev:
            cur_lim /= 2.
        if cur_lim <= 2 ** (-3):
            print('Tiny limit')
            return np.zeros_like(dw), cur_lim, False

    return lim_dw, cur_lim, True


def local_limit(
        Wpq: np.ndarray,
        g_pq: np.ndarray,
        images: Images | ResampledImgs
):
    """
    limit(s) =  2 * sqrt(2) * sigma(s)
    where we calculate sigma based on
    c_pq(s) = sum_pq W_pq(s) * g_pq(s) ** 2 / sum_pq W_pq(s)

    The limit is then the average of these c_pq(s) values across the scales
    and across the views.
    weighted by sigma_q / mag_Bp

    We then multiply the limit by 2 *sqrt(2)

    Parameters
    ----------
    Wpq
    g_pq
    images

    Returns
    -------

    """
    n_tgts, n_res, height, width = Wpq.shape
    sum_weights = np.sum(Wpq * g_pq ** 2, axis=(0, 1))
    # sum_weights = np.sum(Wpq, axis=(0, 1))
    lim = np.zeros((height, width))
    for p in range(n_tgts):
        B_p = images.baselines[p + 1]
        mag_Bp = np.sqrt(B_p[0] ** 2 + B_p[1] ** 2)
        # mag_Bp = 1.0
        for q in range(n_res):
            c_pq = (Wpq[p, q] * g_pq[p, q] ** 2 + 1e-10) / (sum_weights + 1e-10)
            # c_pq = (Wpq[p, q] + 1e-4) / (sum_weights + 1e-4)
            sigma_q = simple.get_sigma(q)
            lim += c_pq * sigma_q / mag_Bp
    lim = lim * 2 * np.sqrt(2)
    return lim


# def local_limit(Wpq: np.ndarray):
#     """
#     limit(s) =  2 * sqrt(2) * sigma(s)
#     where we calculate sigma based on
#     sigma(s) = sum_pq W_pq(s) sigma_q / sum_pq W_pq(s)
#     Parameters
#     ----------
#     Wpq
#
#     Returns
#     -------
#
#     """
#     n_tgts, n_res, height, width = Wpq.shape
#     num = 0
#     den = np.sum(Wpq)
#     for q in range(n_res):
#         sigma_q = 2 * 2 ** q / 3.
#         num += np.sum(Wpq[:, q] * sigma_q)
#     sigma = num / den
#     lim = np.ones((height, width)) * 2 * np.sqrt(2) * sigma
#     return lim


def stability_check_warp(
        images: Images,
        w: np.ndarray,
        dw: np.ndarray,
        state: State,
        context: Context
):
    """
    Repeatedly check the prediction error with at different limits until the
    prediction error goes down. As soon as this happens, we return the limited
    disparity.

    If the prediction error never goes down, we return False or something?

    Parameters
    ----------
    images
    w
    dw
    state
    context

    Returns
    -------

    """
    max_disp = state.disp_lim
    temp_state = copy.deepcopy(state)
    n_iterations = int(np.floor(np.log2(max_disp))) + 1
    prev_err = images.prediction_err(w)
    for n in range(n_iterations):
        print(max_disp)
        lim_disp, limited = limit_disparity(dw, temp_state, context)
        pred_err = images.prediction_err(w + lim_disp)
        if pred_err > prev_err:
            max_disp /= 2
            temp_state.disp_lim = max_disp
        else:
            return lim_disp, limited

    return np.zeros_like(w), False
