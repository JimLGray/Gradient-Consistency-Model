from data_io import fileIO, plot, output
import splines
import numpy as np
from context_class import Context, State
import regex
from img import Images, RawImages, ResampledImgs
from scipy import ndimage
from copy import deepcopy
import multiview as mv
import h5py
import prog_inc as pi
import multiprocessing as mp


def main():
    # fileIO.open_hci_2013('data/training/buddha/lf.h5',
    #                      'data/training/buddha/imgs/')
    # ctf_old.w_init = np.load('output/training/buddha/reference/Disparity_0.npy')

    scenes = [
        # Old dataset
        # 'buddha',
        # 'buddha2',
        # 'horses',
        # 'medieval',
        # 'monasRoom',
        # 'papillon',
        # 'stillLife',

        # New Dataset
        # 'boxes',
        # 'cotton',
        # 'dino',
        'sideboard',

        # 'antinous',
        # 'boardgames',
        # 'dishes',
        # 'greek',
        # 'kitchen',
        # 'medieval2',
        # 'museum',
        # 'pens',
        # 'platonic',
        # 'pillows',
        # 'rosemary',
        # 'table',
        # 'tomb',
        # 'tower',
        # 'town',
        # 'vinyl',

        # Middlebury 2006
        # 'Aloe',
        # 'Baby1',
        # 'Baby2',
        # 'Baby3',
        # 'Bowling1',
        # 'Bowling2',
        # 'Cloth1',
        # 'Cloth2',
        # 'Cloth3',
        # 'Cloth4',
        # 'Flowerpots',
        # 'Lampshade1',
        # 'Lampshade2',
        # 'Midd1',
        # 'Midd2',
        # 'Monopoly',
        # 'Plastic',
        # 'Rocks1',
        # 'Rocks2',
        # 'Wood1',
        # 'Wood2',
    ]

    param_sets = [
        'old',
        # 'warp_free',
        # 'inter_irls',
        # 'grad_const',
        'gc_model',
        # 'wf_gc',
        'gc_no_ge',
        'gc_no_cse',
    ]
    cons = []
    for scene in scenes:
        for param_set in param_sets:
            con = create_context(param_set, scene)
            cons.append(con)

    # with mp.Pool(processes=min(6, len(cons))) as pool:
    #     pool.map(mv.mv_disparity, cons)
    # for con in cons:
    #     mv.mv_disparity(con)

    for scene in scenes:
        title = scene + ' RMSE vs Ax=b solves'
        fname = 'output/plots_macro/' + scene + '.png'

        cons = []
        for param_set in param_sets:
            con = create_context(param_set, scene)
            cons.append(con)

        plot.rmse_vs_matrix_solves_plot(
            cons,
            # constant=[0.15779407, 0.0765856],
            title=title,
            legend=[
                'Traditional Disparity Estimation',
                # 'Warping Free',
                # 'Interpolation in IRLS and Warping Free',
                'Gradient Consistency',
                # 'GC No Confidence Metric',
                # 'Gradient Consistency with Warp Free',
                'Gradient Consistency with No Gradient Error',
                'Gradient Consistency with No Coarse Scale Error',
                # 'Reference (0th warp)',
                # 'Reference (1st warp)'
            ],
    #         save=True,
            filename=fname
        )
    # table_name = 'output/training2/' + 'table.csv'
    # output.res_table(cons, table_name)

def create_context(param_set='old', scene='buddha'):
    dir_name = 'data/training/' + scene + '/' # + 'imgs/'
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

    # Old HCI Light Field Dataset

    # filenames = [
    #     dir_name + '(0, 0).png',
    #     dir_name + '(0, -1).png',
    #     dir_name + '(0, -2).png',
    #     # dir_name + '(0, -3).png',
    #     # dir_name + '(0, -4).png',
    #     dir_name + '(0, 1).png',
    #     dir_name + '(0, 2).png',
    #     # dir_name + '(0, 3).png',
    #     # dir_name + '(0, 4).png',
    #     dir_name + '(1, -1).png',
    #     dir_name + '(1, -2).png',
    #     # dir_name + '(1, -3).png',
    #     # dir_name + '(1, -4).png',
    #     dir_name + '(1, 0).png',
    #     dir_name + '(1, 1).png',
    #     dir_name + '(1, 2).png',
    #     # dir_name + '(1, 3).png',
    #     # dir_name + '(1, 4).png',
    #     dir_name + '(2, -1).png',
    #     dir_name + '(2, -2).png',
    #     # dir_name + '(2, -3).png',
    #     # dir_name + '(2, -4).png',
    #     dir_name + '(2, 0).png',
    #     dir_name + '(2, 1).png',
    #     dir_name + '(2, 2).png',
    #     # dir_name + '(2, 3).png',
    #     # dir_name + '(2, 4).png',
    #     # dir_name + '(3, -1).png',
    #     # dir_name + '(3, -2).png',
    #     # dir_name + '(3, -3).png',
    #     # dir_name + '(3, -4).png',
    #     # dir_name + '(3, 0).png',
    #     # dir_name + '(3, 1).png',
    #     # dir_name + '(3, 2).png',
    #     # dir_name + '(3, 3).png',
    #     # dir_name + '(3, 4).png',
    #     # dir_name + '(4, -1).png',
    #     # dir_name + '(4, -2).png',
    #     # dir_name + '(4, -3).png',
    #     # dir_name + '(4, -4).png',
    #     # dir_name + '(4, 0).png',
    #     # dir_name + '(4, 1).png',
    #     # dir_name + '(4, 2).png',
    #     # dir_name + '(4, 3).png',
    #     # dir_name + '(4, 4).png',
    #     dir_name + '(-1, -1).png',
    #     dir_name + '(-1, -2).png',
    #     # dir_name + '(-1, -3).png',
    #     # dir_name + '(-1, -4).png',
    #     dir_name + '(-1, 0).png',
    #     dir_name + '(-1, 1).png',
    #     dir_name + '(-1, 2).png',
    #     # dir_name + '(-1, 3).png',
    #     # dir_name + '(-1, 4).png',
    #     dir_name + '(-2, -1).png',
    #     dir_name + '(-2, -2).png',
    #     # dir_name + '(-2, -3).png',
    #     # dir_name + '(-2, -4).png',
    #     dir_name + '(-2, 0).png',
    #     dir_name + '(-2, 1).png',
    #     dir_name + '(-2, 2).png',
    #     # dir_name + '(-2, 3).png',
    #     # dir_name + '(-2, 4).png',
    #     # dir_name + '(-3, -1).png',
    #     # dir_name + '(-3, -2).png',
    #     # dir_name + '(-3, -3).png',
    #     # dir_name + '(-3, -4).png',
    #     # dir_name + '(-3, 0).png',
    #     # dir_name + '(-3, 1).png',
    #     # dir_name + '(-3, 2).png',
    #     # dir_name + '(-3, 3).png',
    #     # dir_name + '(-3, 4).png',
    #     # dir_name + '(-4, -1).png',
    #     # dir_name + '(-4, -2).png',
    #     # dir_name + '(-4, -3).png',
    #     # dir_name + '(-4, -4).png',
    #     # dir_name + '(-4, 0).png',
    #     # dir_name + '(-4, 1).png',
    #     # dir_name + '(-4, 2).png',
    #     # dir_name + '(-4, 3).png',
    #     # dir_name + '(-4, 4).png'
    # ]

    # filenames = [
    #     dir_name + '(0, 0).png',
    #     dir_name + '(1, 0).png',
    #     dir_name + '(2, 0).png',
    #     dir_name + '(3, 0).png',
    #     dir_name + '(4, 0).png',
    #     dir_name + '(-1, 0).png',
    #     dir_name + '(-2, 0).png',
    #     dir_name + '(-3, 0).png',
    #     dir_name + '(-4, 0).png',
    #     dir_name + '(0, 1).png',
    #     dir_name + '(0, 2).png',
    #     dir_name + '(0, 3).png',
    #     dir_name + '(0, 4).png',
    #     dir_name + '(0, -1).png',
    #     dir_name + '(0, -2).png',
    #     dir_name + '(0, -3).png',
    #     dir_name + '(0, -4).png',
    # ]
    # baselines, valid_names = regex.regex_2013_lf_data(filenames)

    sorted_baselines, sorted_names = regex.sort_baselines_names(baselines,
                                                                valid_names)
    # dir_name = 'data/middlebury_2006/' + scene + '/'
    # sorted_names = [
    #     dir_name + 'view1.png',
    #     dir_name + 'view0.png',
    #     dir_name + 'view2.png',
    #     dir_name + 'view3.png',
    #     dir_name + 'view4.png',
    #     dir_name + 'view5.png',
    #     dir_name + 'view6.png'
    # ]
    # sorted_baselines = np.array([
    #     [0, 0],
    #     [0, -1],
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [0, 5],
    # ])

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
        con.irls_stages = 50
        con.n_levels = 3
        con.multi_res_levels = 1
        con.total_max_warps = 3
        con.n_warps_list = [1, 1, 1, 1, 1, 1]

        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        alpha = 0.002
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
        con.irls_stages = 50
        con.n_levels = 3
        con.multi_res_levels = 1
        con.total_max_warps = 3
        con.n_warps_list = [1, 1, 1, 1, 1, 1]
        # alpha = 7.5 * (len(sorted_baselines) - 1) / 24
        alpha = 0.002
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

    elif param_set == 'grad_const':
        con.irls_stages = 5
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        # alpha = 7.8125e-06
        alpha = 0.002
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.reg_normalisation = True
        con.confidence_metric = True

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.median_filter = True
        con.apply_limit = False

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False

    elif param_set == 'gc_model':
        con.irls_stages = 50
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        # alpha = 7.8125e-06
        alpha = 0.002
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.reg_normalisation = True
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.median_filter = True
        con.apply_limit = True
        # con.tol = 1e-2

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False

    elif param_set == 'gc_no_ge':
        con.irls_stages = 50
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 1
        con.n_warps_list = [1]
        # alpha = 7.8125e-06
        alpha = 0.002
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.reg_normalisation = True
        con.confidence_metric = False

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.median_filter = True
        con.apply_limit = True

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False
        con.gc_err = False

    elif param_set == 'gc_no_cse':
        con.irls_stages = 50
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 1
        con.n_warps_list = [1, 1]
        # alpha = 7.8125e-06
        alpha = 0.002
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.reg_normalisation = True
        con.confidence_metric = True

        con.interpolate_in_IRLS = True
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = False
        con.median_filter = True
        con.apply_limit = False

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False
        con.coarse_err = False

    elif param_set == 'wf_gc':
        con.irls_stages = 25
        con.n_levels = 1
        con.multi_res_levels = 3
        con.total_max_warps = 15
        con.n_warps_list = [15]
        alpha = 0.005
        con.current_reg_const = alpha
        con.save_weights = False
        con.snr_weights = False
        con.reg_normalisation = True
        con.confidence_metric = True

        con.interpolate_in_IRLS = False
        con.grad_err = True
        con.normalise_opposites = True
        con.normalise_views = False
        con.valid_check = True
        con.median_filter = True
        con.apply_limit = False
        con.tol = 1e-2

        con.stability_check = False
        con.prog_inc = False
        con.raw_imgs = False
        con.outside_image = False

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
    # gt_depth_name = 'data/training/' + scene + '/gt_depth_lowres.pfm'
    # gt_depth, _ = fileIO.readPFM(gt_depth_name)

    # f = h5py.File('data/training/buddha/lf.h5')
    # dH = f.attrs['dH'][0]
    # focal_length = f.attrs['focalLength'][0]
    # delta_x = f.attrs['shift']
    # con.focal_length = focal_length
    # con.delta_x = delta_x
    # con.dH = dH

    # gt_name = 'data/middlebury_2006/' + scene + '/disp1.png'

    # gt_disp = fileIO.open_middlebury2006_gt(gt_name)
    # plot.colorbar_img_plot(
    #     gt_disp, 'GT Disparity',
    #     # vmin=vmin, vmax=vmax
    # )

    con.gt_disparity = gt_disp
    # con.gt_depth = gt_depth

    suffix = param_set + '/' + 'alpha=' + str(alpha) + '/'

    con.res_dir = 'output/training/' + scene + '/' + suffix

    return con


if __name__ == "__main__":
    main()
