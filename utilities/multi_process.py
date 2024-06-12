"""
Runs several multiview disparity calculations at once.
"""

from context_class import Context
from data_io import fileIO, output
import prog_inc as pi
import regex
import multiprocessing as mp
import numpy as np
from copy import deepcopy
from pathlib import Path
import multiview as mv


def main():
    # scenes = ['buddha', 'buddha2', 'horses', 'medieval', 'monasRoom',
    #           'papillon', 'stillLife']
    # scenes = [
    #     'boxes',
    #     'cotton',
    #     'dino',
    #     'sideboard',
    # ]
    # scenes = [
    #     'antinous',
    #     'boardgames',
    #     'dishes', 'greek',
    #     'kitchen',
    #     'medieval2', 'museum', 'pens', 'pillows',
    #     'platonic',
    #     'rosemary',
    #     'table', 'tomb',
    #     'tower',
    #     'town',
    #     'vinyl'
    # ]
    scenes = [
        'Aloe',
        'Baby1',
        'Baby2',
        'Baby3',
        'Bowling1',
        'Bowling2',
        'Cloth1',
        'Cloth2',
        'Cloth3',
        'Cloth4',
        'Flowerpots',
        'Lampshade1',
        'Lampshade2',
        'Midd1',
        'Midd2',
        'Monopoly',
        'Plastic',
        'Rocks1',
        'Rocks2',
        'Wood1',
        'Wood2',
    ]
    ctxes = []
    alphas = [
        # 0.0005,
        # 0.001,
        # 0.002,
        # 0.005,
        # 0.01,
        # 0.02,
        # 0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        # 10.0
    ]

    for scene in scenes:
        for alpha in alphas:
            # old_con = create_context(alpha=alpha, scene=scene,
            #                          param_set='old')
            # Path(old_con.res_dir).mkdir(parents=True, exist_ok=True)
            # ctxes.append(old_con)
            ge_no_con = create_context(alpha=alpha, scene=scene,
                                       param_set='gc_model')
            Path(ge_no_con.res_dir).mkdir(parents=True, exist_ok=True)
            ctxes.append(ge_no_con)
            # gc_no_grad_err = create_context(alpha=alpha,
            #                                 scene=scene,
            #                                 param_set='gc_no_ge')
            # Path(gc_no_grad_err.res_dir).mkdir(parents=True, exist_ok=True)
            # ctxes.append(gc_no_grad_err)
            # gc_no_coarse_err = create_context(alpha=alpha,
            #                                 scene=scene,
            #                                 param_set='gc_no_cse')
            # Path(gc_no_coarse_err.res_dir).mkdir(parents=True, exist_ok=True)
            # ctxes.append(gc_no_coarse_err)

    # with mp.Pool(processes=min(6, len(ctxes))) as pool:
    #     pool.map(mv.mv_disparity, ctxes)
    # for ctx in ctxes:
    #     mv.mv_disparity(ctx)
    table_name = 'output/cse_sum/5_solves/middlebury_2006/' + \
                 'table.csv'
    output.res_table(ctxes, table_name)


def create_context(param_set='old', scene='sideboard', alpha=None):
    # dir_name = 'data/training/' + scene + '/'  # + 'imgs/'
    # # New Light Field Dataset
    # filenames = [
    #     dir_name + 'input_Cam040.png',
    #     dir_name + 'input_Cam004.png',
    #     dir_name + 'input_Cam013.png',
    #     dir_name + 'input_Cam022.png',
    #     dir_name + 'input_Cam031.png',
    #     dir_name + 'input_Cam036.png',
    #     dir_name + 'input_Cam037.png',
    #     dir_name + 'input_Cam038.png',
    #     dir_name + 'input_Cam039.png',
    #     dir_name + 'input_Cam041.png',
    #     dir_name + 'input_Cam042.png',
    #     dir_name + 'input_Cam043.png',
    #     dir_name + 'input_Cam044.png',
    #     dir_name + 'input_Cam049.png',
    #     dir_name + 'input_Cam058.png',
    #     dir_name + 'input_Cam067.png',
    #     dir_name + 'input_Cam076.png'
    # ]
    # baselines, valid_names = regex.regex_light_field(filenames)
    #
    # sorted_baselines, sorted_names = regex.sort_baselines_names(baselines,
    #                                                             valid_names)
    dir_name = 'data/middlebury_2006/' + scene + '/'
    sorted_names = [
        dir_name + 'view1.png',
        dir_name + 'view0.png',
        dir_name + 'view2.png',
        dir_name + 'view3.png',
        dir_name + 'view4.png',
        dir_name + 'view5.png',
        dir_name + 'view6.png'
    ]
    sorted_baselines = np.array([
        [0, 0],
        [0, -1],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
    ])

    con = Context()

    con.data_sigma = 0.01
    con.reg_sigma = 1.0
    con.dynamic_d_sig = True
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

    con.reg_normalisation = True
    con.warp_top_level = True
    con.four_neighbours = True
    con.successive_filtering = False
    con.warp_boundary = 'edge'
    con.warp_mode = 'cubic'
    con.quadratic_acc = False
    con.radial_distortion = -0.125
    con.baseline_noise = 0.0

    if param_set == 'old':
        con.irls_stages = 50
        con.n_levels = 6
        con.multi_res_levels = 1
        con.total_max_warps = 6
        con.n_warps_list = [1, 1, 1, 1, 1, 1]

        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        if alpha is None:
            alpha = 0.002
        con.current_reg_const = alpha
        con.save_weights = False
        con.interpolate_in_IRLS = True
        con.stability_check = False
        con.raw_imgs = True
        con.outside_image = True
        con.grad_err = False
        con.normalise_opposites = False
        con.valid_check = False
        con.apply_limit = True

    elif param_set == 'prog_inc':
        con.irls_stages = 50
        con.n_levels = 3
        con.multi_res_levels = 1
        con.total_max_warps = 5
        con.n_warps_list = [1, 1, 1, 1, 1, 1]
        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        if alpha is None:
            alpha = 0.001
        con.current_reg_const = alpha
        con.reg_normalisation = True
        con.save_weights = False

        con.interpolate_in_IRLS = True
        con.stability_check = False
        con.raw_imgs = False
        con.outside_image = True
        con.grad_err = False
        con.normalise_opposites = False
        con.valid_check = False
        con.apply_limit = True
        con.prog_inc = True
        con.stages = 4
        con.start_n = 5
        con.increment = 4

    elif param_set == 'gc_model':
        con.irls_stages = 10
        con.n_levels = 1
        con.multi_res_levels = 6
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        # alpha = 7.8125e-06
        if alpha is None:
            alpha = 0.005
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        # reg normalisation is off for testing
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.apply_limit = True

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = True
        con.outside_image = True

    elif param_set == 'hybrid':
        con.irls_stages = 50
        con.n_levels = 2
        con.multi_res_levels = 3
        con.total_max_warps = 2
        con.n_warps_list = [1, 1]
        # alpha = 7.8125e-06
        if alpha is None:
            alpha = 0.005
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.apply_limit = True

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False

    elif param_set == 'gc_no_ge':
        con.irls_stages = 50
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        if alpha is None:
            alpha = 0.0005
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.apply_limit = True

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = True
        con.outside_image = False
        con.gc_err = False

    elif param_set == 'gc_no_cse':
        con.irls_stages = 50
        con.n_levels = 1
        con.multi_res_levels = 6
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        if alpha is None:
            alpha = 0.0005
        con.current_reg_const = alpha
        con.save_weights = True
        con.snr_weights = False
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.apply_limit = False

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = True
        con.outside_image = False
        con.coarse_err = False
    else:
        raise ValueError('Invalid param_set')
    con.noise_epsilon = 1e-4  # Assume we're accurate to 1/1000th of the image

    # vmax, vmin = fileIO.read_vmin_vmax(dir_name + 'parameters.cfg')
    vmax = 2 ** 5
    vmin = -(2 ** 5)
    con.vmax = vmax
    con.vmin = vmin

    # gt_name = 'data/training/' + scene + '/gt_disp_lowres.pfm'
    # gt_disp, _ = fileIO.readPFM(gt_name)
    # gt_depth_name = 'data/training/' + scene + '/gt_depth_lowres.pfm'
    # gt_depth, _ = fileIO.readPFM(gt_depth_name)

    gt_name = 'data/middlebury_2006/' + scene + '/disp1.png'
    gt_disp = fileIO.open_middlebury2006_gt(gt_name)

    con.gt_disparity = gt_disp
    # con.gt_depth = gt_depth

    suffix = param_set + '/'

    con.res_dir = 'output/cse_sum/5_solves/middlebury_2006/' \
                  + scene + '/' + suffix +\
                  'alpha=' + str(alpha) + '/'

    return con


if __name__ == "__main__":
    main()
