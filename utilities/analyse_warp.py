import numpy as np
from data_io import plot, fileIO
from scipy import io
import regex
from img import RawImages, TopScaleSplines, GaussianSplines
from context_class import Context, State
from PIL import Image
import multiview as mv


def create_context(param_set='old', scene='buddha'):
    dir_name = 'data/training/' + scene + '/'  # + 'imgs/'
    # New Light Field Dataset
    filenames = [
        dir_name + 'input_Cam040.png',
        dir_name + 'input_Cam004.png',
        dir_name + 'input_Cam013.png',
        dir_name + 'input_Cam022.png',
        dir_name + 'input_Cam031.png',
        dir_name + 'input_Cam036.png',
        dir_name + 'input_Cam037.png',
        dir_name + 'input_Cam038.png',
        dir_name + 'input_Cam039.png',
        dir_name + 'input_Cam041.png',
        dir_name + 'input_Cam042.png',
        dir_name + 'input_Cam043.png',
        dir_name + 'input_Cam044.png',
        dir_name + 'input_Cam049.png',
        dir_name + 'input_Cam058.png',
        dir_name + 'input_Cam067.png',
        dir_name + 'input_Cam076.png'
    ]
    baselines, valid_names = regex.regex_light_field(filenames)

    sorted_baselines, sorted_names = regex.sort_baselines_names(baselines,
                                                                valid_names)
    con = Context()

    con.data_sigma = 0.1
    con.dynamic_d_sig = True
    con.reg_sigma = 1.0  # this doesn't really make much sense...
    con.data_cost_func = 'L1'
    con.reg_cost_func = 'L1'

    con.filenames = sorted_names
    con.baselines = np.array(sorted_baselines)
    con.max_iter = 512 * 512
    con.median_filter = True
    if con.data_cost_func == 'Welsch':
        con.grad_scaling = False
        con.view_scaling = False
    else:
        con.grad_scaling = False
        con.view_scaling = False

    con.warp_top_level = True
    con.four_neighbours = True
    con.successive_filtering = False
    con.baseline_normalisation = False
    con.warp_boundary = 'edge'
    con.warp_mode = 'cubic'
    con.quadratic_acc = False
    con.apply_limit = False

    if param_set == 'old':
        con.irls_stages = 5
        con.n_levels = 3
        con.multi_res_levels = 1
        con.total_max_warps = 3
        con.n_warps_list = [1, 1, 1, 1, 1, 1]

        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        alpha = 0.005
        con.current_reg_const = alpha
        con.interpolate_in_IRLS = True
        con.stability_check = False
        con.raw_imgs = True
        con.outside_image = True
        con.grad_err = False
        con.normalise_opposites = False
        con.valid_check = False
        con.apply_limit = True
        con.reg_normalisation = True

    elif param_set == 'inter_irls':
        con.irls_stages = 5
        con.n_levels = 3
        con.multi_res_levels = 1
        con.total_max_warps = 3
        con.n_warps_list = [1, 1, 1, 1, 1, 1]
        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        alpha = 0.005
        con.current_reg_const = alpha
        con.reg_normalisation = True

        con.interpolate_in_IRLS = True
        con.stability_check = False
        con.raw_imgs = False
        con.outside_image = True
        con.grad_err = False
        con.normalise_opposites = False
        con.valid_check = False
        con.apply_limit = True


    else:
        raise ValueError('Invalid param_set')
    con.noise_epsilon = 1e-4

    vmin, vmax = fileIO.read_vmin_vmax(dir_name + 'parameters.cfg')
    # vmax = 2 ** 5
    # vmin = -(2 ** 5)
    con.vmax = vmax
    con.vmin = vmin

    gt_name = 'data/training/' + scene + '/gt_disp_lowres.pfm'
    gt_disp, _ = fileIO.readPFM(gt_name)
    con.gt_disparity = gt_disp

    # plot.colorbar_img_plot(
    #     gt_disp, 'GT Disparity',
    #     # vmin=vmin, vmax=vmax
    # )
    suffix = param_set + '/'  # + 'alpha=' + str(alpha) + '/'
    con.res_dir = 'output/scratch/warp_test/' + scene + '/' + suffix

    return con


def patch_view(data: np.ndarray, title: str, out_dir: str, top_left: tuple,
               bottom_right: tuple,
               bases_list: list, vmax=1.0, vmin=0.0):
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
                 top_left[1]: bottom_right[1]] * 255
            title_str = title + str(bases_list[p]) + ' Scale: ' + str(q)
            out_name = out_dir + title_str + '.png'
            # int_im = im.astype('int')
            save_img = Image.fromarray(im)
            save_img = save_img.convert('L')
            save_img.save(out_name)



scenes = [
    'sideboard',
]

param_sets = [
    'inter_irls',
    'old'
]
cons = []
for scene in scenes:
    for param_set in param_sets:
        con = create_context(param_set, scene)
        cons.append(con)

window_tl = (0, 0)
window_br = (512, 512)

baselines = [(0, 0), (1, 0), (0, -1), (0, 1), (-1, 0), (2, 0), (0, -2),
             (0, 2),
             (-2, 0), (3, 0), (0, -3), (0, 3), (-3, 0), (4, 0), (0, -4),
             (0, 4),
             (-4, 0)]

tgt_bases = [(1, 0), (0, -1), (0, 1), (-1, 0), (2, 0), (0, -2),
             (0, 2),
             (-2, 0), (3, 0), (0, -3), (0, 3), (-3, 0), (4, 0), (0, -4),
             (0, 4),
             (-4, 0)]

for con in cons:
    state = State()
    state.vmin = con.vmin
    state.cur_baselines = con.baselines
    state.vmax = con.vmax
    w = con.gt_disparity
    if con.raw_imgs:
        images = RawImages(con, state)
    else:
        images = GaussianSplines(con, state)

    images.save_warped_imgs(con.res_dir, w, 0)
    img_height = images.height
    img_width = images.width
    max_q = images.n_res
    n_targets = images.n_targets

    dot_grads = np.empty((n_targets, max_q, img_height, img_width))
    raw_grads = np.empty((n_targets + 1, max_q, img_height, img_width, 2))
    diff = np.empty_like(dot_grads)

    mv.update_data(images, w, 0, raw_grads, dot_grads, diff)

    views = np.load(con.res_dir + 'warped_imgs_0.npy')
    v_title = 'View (y, x): '
    patch_view(views, v_title, con.res_dir, window_tl, window_br, baselines)

    d_title = 'Diff (y, x):'
    patch_view(np.abs(diff), d_title, con.res_dir, window_tl, window_br,
               tgt_bases)

    g_title = 'Grad (y, x):'
    patch_view(np.abs(dot_grads), g_title, con.res_dir, window_tl, window_br,
               tgt_bases)

