# Module prog_inc.py

"""
Performs a progressive inclusion of views. Basically calls mv.mv_disparity
multiple times with different numbers of views with various sets of parameters.
"""
import numpy as np
from context_class import Context, State
from img import RawImages
import multiview as mv
import weights
from graph import FullWeightedGraph
import cgrad as cg
from scipy import ndimage
from data_io import output as out


def prog_inc(context: Context):
    """
    Performs progressive inclusion of views given the parameters in context at
    a single scale only. We use only an unweighted approach.

    Parameters
    ----------
    context

    Returns
    -------

    """
    if context.w_init is None:
        w = np.zeros_like(context.gt_disparity, dtype=float)
    else:
        w = context.w_init
    n_stages = context.stages
    irls_stages = context.irls_stages
    height = context.gt_disparity.shape[0]
    width = context.gt_disparity.shape[1]
    reg_sigma = context.reg_sigma
    state = State()

    for n in range(n_stages):
        n_imgs = (context.start_n + n * context.increment) ** context.power
        n_total_imgs = len(context.baselines)
        filenames = context.filenames[0:n_imgs]
        baselines = context.baselines[0:n_imgs]
        alpha = context.current_reg_const * n_imgs / n_total_imgs
        temp_con = context.copy()
        temp_con.filenames = filenames
        temp_con.baselines = baselines
        state.cur_baselines = baselines
        images = RawImages(temp_con, state)
        n_targets = images.n_targets
        max_q = images.n_res
        assert max_q == 1

        dot_grads = np.empty((n_targets, max_q, height, width))
        raw_grads = np.empty((n_targets + 1, max_q, height, width, 2))
        diff = np.empty_like(dot_grads)
        print('Progressive Inclusion Stage: ', state.prog_inc_stage)
        print(n_imgs)
        state.update_limit()
        print(state.disp_lim)

        for i in range(irls_stages):
            print('Ax = b Solves:', i)
            state.update_irls_stage(i)
            dw = np.zeros_like(w)
            mv.update_data(images, w, 0, raw_grads, dot_grads, diff)
            scale = mv.update_scale(context, raw_grads, n_targets)

            if context.dynamic_d_sig and context.data_cost_func == 'Welsch':
                state.calculate_sig_d(dot_grads, diff, w)
            else:
                state.current_d_sig = context.data_sigma

            if context.data_cost_func == 'L2' or state.is_first_stage():
                irls_weights = np.ones_like(diff)
            else:
                irls_weights = weights.robust_weights(
                    dot_grads, diff, dw, scale, context, state
                )
            total_weights = irls_weights * scale
            sum_g_pq = np.sum(
                dot_grads ** 2 * total_weights, axis=(0, 1)
            )
            sum_delta_Ipq = np.sum(
                -diff * dot_grads * total_weights, axis=(0, 1)
            )

            reg = FullWeightedGraph(height, width, sig_r=reg_sigma)

            if context.reg_cost_func != 'L2' and not state.is_first_stage():
                if context.reg_cost_func == 'L1' and context.four_neighbours:
                    reg.set_weights_4_neighbours_l1(w + dw, 0)
                else:
                    reg.set_weights(w + dw, reg_sigma, context.reg_cost_func)

            reg.apply_w_kernel(alpha, context.four_neighbours)
            dw = cg.cg_disparity(
                w, dw, sum_g_pq, sum_delta_Ipq, reg, context.max_iter,
                context.tol
            )
            if context.apply_limit:
                dw, is_limited = mv.limit_disparity(dw, state, context)
                if state.prog_inc_stage == context.stages - 1:
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
            if continue_irls is False:
                break

        state.prog_inc_stage += 1
        state.update_warp_stage(state.prog_inc_stage)
    return w
