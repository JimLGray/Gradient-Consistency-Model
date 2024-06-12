import json

from data_io import fileIO, plot
import numpy as np
import regex
from pathlib import Path

# coordinates in (y, x) cause that's how numpy arrays work

window_tl = (0, 0)
window_br = (512, 512)
# window_br = (577, 641)
# window_tl = (150, 475)
# window_br = (175, 500)
# window_tl = (140, 40)
# window_br = (200, 100)

folder = 'output/scratch/additional/antinous/gc_model/alpha=0.2/'

fp = open(folder + 'context_0.json')

baselines = [(0, 0), (1, 0), (0, -1), (0, 1), (-1, 0), (2, 0), (0, -2),
             (0, 2),
             (-2, 0), (3, 0), (0, -3), (0, 3), (-3, 0), (4, 0), (0, -4),
             (0, 4),
             (-4, 0)]


# baselines = np.array([
#     [0, 0],
#     # [0, -1],
#     [0, 1],
#     [0, 2],
#     # [0, 3],
#     # [0, 4],
#     # [0, 5],
# ])

def patch_view(data: np.ndarray, title: str, out_dir: str, top_left: tuple,
               bottom_right: tuple,
               bases_list: list, vmax=None, vmin=None):
    """
    Saves a patch of data which is originally a .npy array as a series of images
    in out_dir

    Parameters
    ----------
    data
    title
    out_dir
    top_left
    bottom_right
    bases_list

    Returns
    -------

    """
    n_imgs = data.shape[0]
    n_res = data.shape[1]

    for p in range(n_imgs):
        for q in range(n_res):
            im = data[p, q, top_left[0]: bottom_right[0],
                 top_left[1]: bottom_right[1]]
            title_str = title + str(bases_list[p]) + ' Scale: ' + str(q)
            out_name = out_dir + title_str + '.png'
            plot.colorbar_img_plot(im, title_str, save=True, filename=out_name,
                                   cmap='Greys', vmax=vmax, vmin=vmin)


def reg_view(reg_graph: np.ndarray, title: str, out_dir: str, top_left: tuple,
             bottom_right: tuple):
    n_rows = reg_graph.shape[2]
    n_cols = reg_graph.shape[3]
    for r in range(n_rows):
        for c in range(n_cols):
            im = -reg_graph[top_left[0]: bottom_right[0],
                  top_left[1]: bottom_right[1], r, c]
            title_str = title + '(' + str(r) + ',' + str(c) + ')'
            out_name = out_dir + title_str + '.png'
            plot.colorbar_img_plot(im, title_str, save=True, filename=out_name,
                                   cmap='Greys')


for irls_stage in range(50):
    res_dir = folder + 'irls_' + str(irls_stage) + '/'
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    views = np.load(folder + 'warped_imgs_' + str(irls_stage) + '.npy')
    v_title = 'View (y, x): '

    patch_view(views, v_title, res_dir, window_tl, window_br, baselines)

    weights = np.load(folder + 'Weights_0_' + str(irls_stage) + '.npy')
    w_title = 'Weights (y, x): '
    #
    n_targets = weights.shape[0]
    n_res = weights.shape[1]
    final_weights = np.empty_like(weights)
    maxes = np.sum(weights, axis=(0, 1))
    for q in range(n_res):
        for p in range(n_targets):
            # plot.colorbar_img_plot(d_weights[p, q])
            final_weights[p, q] = n_res * n_targets * weights[p, q] / \
                                  (maxes + 1e-16)

    patch_view(final_weights, w_title, res_dir, window_tl, window_br,
               baselines[1:], vmax=16, vmin=0)
    #
    diffs = np.load(folder + 'Diffs_Unweighted_0_' + str(irls_stage) + '.npy')

    diff_title = 'Diff (y, x): '

    grads = np.load(folder + 'Grads_Unweighted_0_' + str(irls_stage) + '.npy')
    grad_title = 'Grad (y, x): '
    patch_view(np.abs(diffs) / np.sqrt(grads), diff_title, res_dir, window_tl,
               window_br, baselines[1:])
    patch_view(np.sqrt(grads), grad_title, res_dir, window_tl, window_br,
               baselines[1:])

    err = np.load(folder + 'Error_0_' + str(irls_stage) + '.npy')
    err_title = 'Error (y, x): '
    patch_view(err, err_title, res_dir, window_tl, window_br, baselines[1:])

    grad_err = np.load(folder + 'Grad_Error_0_' + str(irls_stage) + '.npy')
    grad_err_title = 'Grad Error (y, x): '
    patch_view(grad_err, grad_err_title, res_dir, window_tl, window_br,
               baselines[1:])

    coarse_err = np.load(folder + 'CoarseError_0_' + str(irls_stage) + '.npy')
    coarse_title = 'Coarse Scale Error (y, x): '
    patch_view(coarse_err, coarse_title, res_dir, window_tl, window_br,
               baselines[1:])

    delta_w = np.load(folder + 'Delta_w_0_' + str(irls_stage) + '.npy')
    delta_w_title = 'Delta w (y, x): '
    patch_view(delta_w, delta_w_title, res_dir, window_tl, window_br,
               baselines[1:])
    # reg_graph = np.load(folder + 'reg_graph_0_' + str(irls_stage) + '.npy')
    # reg_title = 'Regulariser Weights: '
    # reg_view(reg_graph, reg_title, res_dir, window_tl, window_br)
    #
    # dweights_sum = np.load(folder + 'sum_Wpq_0_' + str(irls_stage) + '.npy')
    # dweight_title = 'Sum of Data Weights'
    # filename = res_dir + dweight_title + '.png'
    # dweight_win = dweights_sum[window_tl[0]:window_br[0],
    #               window_tl[1]:window_br[1]]
    # plot.colorbar_img_plot(
    #     dweight_win / np.amax(dweight_win),
    #     dweight_title, save=True, filename=filename,
    #     cmap='Greys'
    # )
    #
    disparity = np.load(folder + 'Disparity_0_' + str(irls_stage) + '.npy')
    filename = res_dir + 'Disparity' + '.png'

    plot.colorbar_img_plot(
        disparity[window_tl[0]:window_br[0], window_tl[1]:window_br[1]],
        'Disparity', save=True, filename=filename,
        cmap='Greys'
    )
