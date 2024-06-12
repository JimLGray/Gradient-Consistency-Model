# Module weights.py
from __future__ import annotations

from img import Images, hpf_gaussian, ResampledImgs, RawImages
from context_class import State, Context
import numpy as np
from scipy import ndimage
import simple
import adj_mat as am


def stable_gc_weights(
        imgs: RawImages,
        raw_grads: np.ndarray,
        dot_grads: np.ndarray,
        diff: np.ndarray,
        w: np.ndarray,
        state: State,
        context: Context
):
    """
    Uses the equation

    g^2_{p,q}(s)sigma_q^2
    -----------------------------------------------------------------------------------------------------------
    sigma_q^2 G_{p,q}^2(s)deltaI^2_{p,q}(s) + sigma_q^2 g^2_{p,q}(s) C_{p,q}^2(s) + g^2_{p,q}(s) \epsilon^2 \pi

    Parameters
    ----------
    imgs
    dot_grads
    diff
    w
    state
    context

    Returns
    -------

    """
    height = imgs.height
    width = imgs.width
    n_pairs = imgs.n_targets
    n_res = imgs.n_res

    gc_weights = np.empty((n_pairs, n_res, height, width))
    grad_err = np.zeros_like(gc_weights)
    N = np.zeros_like(gc_weights)
    coarse_err = np.zeros_like(gc_weights)
    w_std = delta_w_schedule(state, w, 3, 2)  # get w_std
    delta_w = np.zeros_like(gc_weights)

    noise_epsilon = imgs.noise_epsilon
    n_sectors = 8

    g_pq_sum = np.sum(dot_grads ** 2, axis=(0, 1))
    delta_I_pq_sum = np.sum(diff ** 2, axis=(0, 1))

    delta_I_0q_sum = np.sum(diff[:, 0, ], axis=0)
    g_0q_sum = np.sum(dot_grads[:, 0], axis=0)
    dw = np.abs(delta_I_0q_sum) / (np.abs(g_0q_sum) + noise_epsilon)
    # dw = ndimage.gaussian_filter(dw_unf, sigma=simple.get_sigma(0))
    # dw = context.gt_disparity - w

    for q in range(n_res):
        ref_q = raw_grads[0, q]
        sector_min = np.full((n_sectors, height, width), np.inf)
        # monotonically decreasing weights in that sector.

        sigma = simple.get_sigma(q)
        for p in range(n_pairs):
            baseline = context.baselines[p + 1]
            B_p = imgs.baselines[p + 1]
            B_mag_sq = B_p[0] ** 2 + B_p[1] ** 2

            sector = am.sector(baseline)
            g_pq = dot_grads[p, q]
            tgt_q = raw_grads[p + 1, q]
            ave_g = (tgt_q + ref_q) / 2.0
            ave_mag_sq = np.sum(ave_g ** 2, axis=-1)

            f_gpq = ndimage.gaussian_filter(g_pq_sum, sigma, mode='nearest')
            delta_Ipq = ndimage.gaussian_filter(delta_I_pq_sum, sigma,
                                                mode='nearest')

            vpq_term = (
                delta_Ipq ** 2 + noise_epsilon ** 2 * np.pi / sigma ** 2) / (
                f_gpq ** 2 + noise_epsilon) + w_std ** 2
            delta_w[p, q] = vpq_term

            if context.gc_err:
                Gpq = (np.sum(B_p * (tgt_q - ref_q), axis=-1) / 2) ** 2
            else:
                Gpq = 0
            if context.coarse_err:
                Cpq = spatial_gc(dot_grads[p, 0], g_pq, dw, q)
            else:
                Cpq = 0
            # The Gpq and Cpq are already squared.

            noise_term = noise_epsilon ** 2 * np.pi * f_gpq ** 2 + 1e-12
            gc_term = Gpq * (
                    delta_Ipq ** 2 * sigma ** 2 + noise_epsilon ** 2 * np.pi)
            cse_term = sigma ** 2 * f_gpq ** 2 * (Gpq * w_std ** 2 + Cpq)

            num = f_gpq ** 2 * sigma ** 2
            den = (noise_term + gc_term + cse_term) #  * (ave_mag_sq + 1e-4)
            c_weights = num / den
            # corrected weights for baseline.
            sector_min[sector] = np.where(c_weights < sector_min[sector],
                                          c_weights, sector_min[sector])
            gc_weights[p, q] = sector_min[sector]

            coarse_err[p, q] = Cpq

            grad_err[p, q] = Gpq
            N[p, q] = coarse_err[p, q] + grad_err[p, q] + \
                      noise_epsilon ** 2 * np.pi / sigma

    return gc_weights, N, grad_err, coarse_err, delta_w


def grad_con(nabla_Ipq: np.ndarray, nabla_I0q: np.ndarray,
             B_mag_sq: float):
    """
    Calculates the gradient consistency for a given pair of images. This
    is essentially a noise term.

    E[Q_pq^2] = | nabla I_pq - nabla I_0q |^2 |B_p|^2 / 4

    Parameters
    ----------
    nabla_Ipq: np.ndarray
        the gradient in the target view
    nabla_I0q: np.ndarray
        the gradient in the reference view
    B_mag_sq: float
        the magnitude of the baseline between both views.

    Returns
    -------
    E[G_pq^2]: np.ndarray
        as per above equation.

    """
    # To save space B_mag, G_pq and g_diff_mag have already been squared.
    g_diff = nabla_Ipq - nabla_I0q
    g_diff_mag = g_diff[:, :, 0] ** 2 + g_diff[:, :, 1] ** 2
    G_pq = g_diff_mag * B_mag_sq / 4
    return G_pq


def delta_w_schedule(state: State, w: np.ndarray, n: int, N: int):
    """
    Scheduled transition between w_upper and w_std. The schedule is a geometric
    series.

    w_upper is derived from the state and w is the disparity field. w_std
    is simply the standard deviation of w over a gaussian window.

    Parameters
    ----------
    state
    w
    n: int
        which stage we are at in the schedule. Goes from 1 to N
    N: int
        Maximum value of n

    Returns
    -------

    """
    w_upper = max(np.abs(state.vmin), np.abs(state.vmax))
    w_var = simple.gaussian_win_var(w, 2. / 3., 'nearest')
    w_var = np.where(w_var > 0, w_var, 0)
    w_std = np.sqrt(w_var)
    ratio = np.power((w_std / w_upper), 1 / (N - 1))
    delta_w = w_upper * np.power(ratio, n - 1)
    if n > N:
        return w_std
    else:
        return delta_w


def spatial_gc_old(
        g_p0: np.ndarray,
        g_pq: np.ndarray,
        dw: np.ndarray,
        q: int,
):
    """
    See the bottom of the Spatial Consistency Analysis v3
    Parameters
    ----------
    g_p0
    g_pq
    dw
    q

    Returns
    -------

    """
    if q == 0:
        # any difference between the quantities makes little to no sense at
        # the top scale
        return 0
    sig = simple.get_sigma(q)
    var_dw = simple.gaussian_win_var(dw, sig, 'reflect')
    var_g_p0 = simple.gaussian_win_var(g_p0, sig, 'reflect')
    var_product = simple.gaussian_win_var(g_p0 * dw, sig, 'reflect')

    res = var_product + var_dw * (g_pq ** 2 + var_g_p0)

    # res = np.where(res < 0, 0, res)
    return res


def spatial_gc(
        g_p0: np.ndarray,
        g_pq: np.ndarray,
        dw: np.ndarray,
        q: int,
):
    """
    See Spatial Consistency Analysis v4

    Parameters
    ----------
    g_p0
    g_pq
    dw
    q

    Returns
    -------

    """
    if q == 0:
        # any difference between the quantities makes little to no sense at
        # the top scale
        return 0
    sig = simple.get_sigma(q)
    var_dw = simple.gaussian_win_var(dw, sig, 'reflect')
    var_gp0 = simple.gaussian_win_var(g_p0, sig, 'reflect')
    O_pq = var_gp0 * var_dw
    return O_pq


def normalise_weights(grads: np.ndarray, d_weights: np.ndarray, images: Images,
                      w: np.ndarray,
                      context: Context):
    """
    Parameters
    ----------
    w
    images
    grads
    d_weights
    context

    Returns
    -------

    """
    # from data_io import plot
    n_targets = d_weights.shape[0]
    n_res = d_weights.shape[1]
    final_weights = np.empty_like(d_weights)
    if context.reg_normalisation:
        maxes = np.sum(d_weights, axis=(0, 1))
        # plot.colorbar_img_plot(maxes)
        for q in range(n_res):
            for p in range(n_targets):
                # plot.colorbar_img_plot(d_weights[p, q])
                final_weights[p, q] = n_res * n_targets * d_weights[p, q] / \
                                      (maxes + 1e-16)

        return final_weights

    u_weights = np.zeros_like(d_weights)  # / n_res
    u_weights[:, 0] = 1.0
    if context.normalise_opposites:
        for sec in range(0, 4):
            opp_sec = sec + 4
            s_indexes = am.get_sector(context.baselines[1:], sec)
            o_indexes = am.get_sector(context.baselines[1:], opp_sec)
            indexes = s_indexes + o_indexes
            if len(indexes) == 0:
                continue

            indexes.sort()

            g_sec = grads[indexes]
            u_sec_weights = u_weights[indexes]
            d_sec_weights = d_weights[indexes]

            u_sum = np.sum(g_sec ** 2 * u_sec_weights, axis=(0, 1))
            d_sum = np.sum(g_sec ** 2 * d_sec_weights, axis=(0, 1))
            ratio = u_sum / (d_sum + 1e-10)

            for q in range(n_res):
                for i in indexes:
                    d_w_iq = d_weights[i, q]
                    if context.valid_check:
                        v_region_p = images.valid_regions(i, w)
                    else:
                        v_region_p = np.ones_like(ratio)
                    final_weights[i, q] = d_w_iq * ratio * v_region_p
    elif context.normalise_views:
        for p in range(n_targets):
            u_view = u_weights[p]
            d_view = d_weights[p]
            g_p = grads[p]
            u_sum = np.sum(g_p ** 2 * u_view, axis=0)
            d_sum = np.sum(g_p ** 2 * d_view, axis=0)
            ratio = u_sum / (d_sum + 1e-10)
            if context.valid_check:
                v_region_p = images.valid_regions(p, w)
            else:
                v_region_p = np.ones_like(ratio)
            for q in range(n_res):
                final_weights[p, q] = d_weights[p, q] * ratio * v_region_p
    else:
        u_sum = np.sum(grads ** 2 * u_weights, axis=(0, 1))
        d_sum = np.sum(grads ** 2 * d_weights, axis=(0, 1))
        ratio = u_sum / (d_sum + 1e-10)

        for p in range(n_targets):
            if context.valid_check:
                v_region_p = images.valid_regions(p, w)
            else:
                v_region_p = np.ones_like(ratio)
            for q in range(n_res):
                d_w_pq = d_weights[p, q]
                final_weights[p, q] = d_w_pq * ratio * v_region_p

    return final_weights


def robust_weights(grads: np.ndarray, diff: np.ndarray,
                   dw: np.ndarray, scale: np.ndarray | float | int,
                   context: Context,
                   state: State):
    """

    Parameters
    ----------
    grads
    diff
    dw:
        the incremental change in w as calculated per irls stage.
    scale:
    context
    state

    Returns
    -------

    """
    if context.dynamic_d_sig:
        sig_x2_sq = 2 * state.current_d_sig ** 2
    else:
        sig_x2_sq = 2 * context.data_sigma ** 2
    height = grads.shape[2]
    width = grads.shape[3]
    n_pairs = grads.shape[0]
    n_res = grads.shape[1]
    delta = 1e-7

    r_weights = np.empty((n_pairs, n_res, height, width))

    for p in range(n_pairs):
        for q in range(n_res):
            g_pq = grads[p, q]
            delta_Ipq = diff[p, q]

            # res = g_pq * dw + delta_Ipq
            if type(scale) == float:
                s = scale
            elif type(scale) == np.ndarray:
                s = scale[p, q]
            sq_gpq = g_pq ** 2 * s
            x_term = g_pq * delta_Ipq * s
            sq_d_Ipq = delta_Ipq ** 2 * s
            res = sq_gpq * dw ** 2 + 2 * x_term * dw + sq_d_Ipq
            if (context.data_cost_func == 'Welsch'
                    or context.data_cost_func == 'welsch'):
                temp = np.exp(-res / sig_x2_sq)
            elif context.data_cost_func == 'L1':
                # den = np.where(np.abs(res) < delta, delta, np.abs(res))
                den = np.sqrt(res + delta)
                temp = 1.0 / den
            else:
                raise ValueError('Unrecognised Robust Cost Function.')
            r_weights[p, q] = temp

    return r_weights


def calc_confidence(grad_err: np.ndarray, coarse_err: np.ndarray,
                    dot_grads: np.ndarray, baselines):
    """

    Parameters
    ----------
    grad_err
    coarse_err
    dot_grads

    Returns
    -------

    """
    confidence = 1 - (grad_err + coarse_err) / (dot_grads ** 2 + 1e-12)
    confidence = np.where(confidence < 0, 0, confidence)

    n_targets = dot_grads.shape[0]
    n_res = dot_grads.shape[1]
    height = dot_grads.shape[2]
    width = dot_grads.shape[3]
    n_sectors = 8

    # apply monotonicity

    for q in range(n_res):
        sector_min = np.full((n_sectors, height, width), np.inf)
        for p in range(n_targets):
            baseline = baselines[p + 1]
            sector = am.sector(baseline)
            confidence[p, q] = np.where(confidence[p, q] < sector_min[sector],
                                        confidence[p, q], sector_min[sector])
            sector_min[sector] = confidence[p, q]
    return confidence
