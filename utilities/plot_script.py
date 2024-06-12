import numpy as np
from matplotlib import pyplot as plt
from data_io import fileIO
from data_io.plot import plot_img_tight


def white_box(img, top, bot, left, right):
    """

    Parameters
    ----------
    img: np.ndarray
    top: int
        top row
    bot: int
        bottom row
    left: int
       left col
    right: int
        right col

    Returns
    -------

    """
    out = np.copy(img)
    out[top - 1:bot + 1, left - 1] = np.nan
    out[top - 1:bot + 1, right + 1] = np.nan
    out[top - 1, left - 1:right + 1] = np.nan
    out[bot + 1, left - 1:right + 1] = np.nan
    return out


def zoom_plot():
    """
    Plots the images and then a zoomed in segment of the images.
    Returns
    -------

    """

    gt_disp, _ = fileIO.readPFM('data/additional/tower/gt_disp_lowres.pfm')
    normal_disparity = np.load(
        'output/additional/tower/gc_no_con/alpha=0.005/Disparity_0_49.npy'
    )
    no_cse = np.load(
        'output/additional/tower/gc_no_coarse_err/alpha=0.005/'
        'Disparity_0_49.npy'
    )

    diff = np.abs(normal_disparity - no_cse)

    vmin = np.amin(gt_disp)
    vmax = np.amax(gt_disp)

    north = 176
    south = north+32
    west = 155
    east = west+32

    gt_box = white_box(gt_disp, north, south, west, east)
    normal_box = white_box(normal_disparity, north, south, west, east)
    cse_box = white_box(no_cse, north, south, west, east)
    diff_box = white_box(diff, north, south, west, east)

    plot_img_tight(gt_box, (2, 4, 1), vmin, vmax)
    plt.title('Ground Truth')
    plot_img_tight(normal_box, (2, 4, 2), vmin, vmax)
    plt.title('GC Full Model')
    plot_img_tight(cse_box, (2, 4, 3), vmin, vmax)
    plt.title('GC No CSE')
    plot_img_tight(diff_box, (2, 4, 4), vmin, vmax)
    plt.title('Diff from No CSE')

    plot_img_tight(
        gt_disp[north:south, west:east], (2, 4, 5), vmin, vmax, ticks=True
    )
    plot_img_tight(
        normal_disparity[north:south, west:east],
        (2, 4, 6),
        vmin,
        vmax,
        ticks=True
    )
    plot_img_tight(
        no_cse[north:south, west:east],
        (2, 4, 7),
        vmin,
        vmax,
        ticks=True
    )

    plot_img_tight(
        diff[north:south, west:east],
        (2, 4, 8),
        vmin, vmax,
        ticks=True
    )

    figure = plt.gcf()
    plt.tight_layout()
    figure.set_size_inches(14, 7)
    plt.subplots_adjust(wspace=0.125, hspace=0., left=0.05, right=0.95,
                        bottom=0.05, top=0.85)

    plt.show()


def plot_n_scenes():
    """
    Plots the disparity fields and the error of n scenes with each method.
    Returns
    -------

    """
    import os
    import re

    scenes = [
        'antinous'
    ]
    methods = [
        'Naive',
        'CTF',
        'Prog. Inc.',
        # 'WF_GC',
        'GC Model']
    method_fnames = [
        'old/',
        'old/',
        'prog_inc/',
        # 'wf_gc/',
        'gc_model/'
    ]

    method_dict = dict(zip(methods, method_fnames))

    training_scenes = ['boxes', 'cotton', 'dino', 'sideboard']

    gt_name = 'gt_disp_lowres.pfm'

    n_scenes = len(scenes)
    n_methods = len(methods)

    scene_idx = 1
    for scene in scenes:

        if scene in training_scenes:
            div_folder = 'training/'
        else:
            div_folder = 'additional/'
        method_idx = 1

        for method in methods:
            if method == 'Naive':
                solve_dir = '2_solves/'
                alpha = 0.5
            elif method == 'GC Model':
                solve_dir = '2_solves/'
                alpha = 0.2
            elif method == 'Prog. Inc.':
                solve_dir = '2_solves/'
                alpha = 0.1
            elif method == 'CTF':
                solve_dir = '2_solves_ctf/'
                alpha = 0.2

            res_scene_path = 'output/' + solve_dir + div_folder + scene + '/'
            gt_scene_path = 'data/' + div_folder + scene + '/' + gt_name

            gt, _ = fileIO.readPFM(gt_scene_path)
            vmin = np.amin(gt)
            vmax = np.amax(gt)

            method_fname = method_dict[method]
            folder_name = res_scene_path + method_fname + 'alpha=' + \
                          str(alpha) + '/'

            file_list = os.listdir(folder_name)
            trim_list = []

            for file in file_list:
                if file[0:9] == 'Disparity':
                    trim_list.append(file)

            max_warp = 0
            max_stage = 0
            pattern = re.compile('\d+')

            for file in trim_list:
                finds = re.findall(pattern, file)
                warp_no = int(finds[0])
                if warp_no >= max_warp:
                    max_warp = warp_no
                    stage = int(finds[1])
                    if stage >= max_stage:
                        max_stage = stage
            print('Scene: ', scene)
            print('   Method: ', method)
            print('      Max Warp:', max_warp)
            print('      Max Stage:', max_stage)
            last_file = 'Disparity_' + str(max_warp) + '_' + str(max_stage) + \
                        '.npy'

            final_disp = np.load(folder_name + last_file)
            diff = np.abs(final_disp - gt)

            disp_pos = (
                n_scenes * 2,
                n_methods,
                (scene_idx - 1) * n_methods * 2 + method_idx
            )
            err_pos = (
                n_scenes * 2,
                n_methods,
                (scene_idx - 1) * n_methods * 2 + n_methods + method_idx
            )
            print(err_pos)

            if method_idx == 1:
                disp_ylabel = scene
                err_ylabel = scene + ' error'
            else:
                disp_ylabel = None
                err_ylabel = None

            if scene_idx == n_scenes:
                xlabel = method
            else:
                xlabel = None

            plot_img_tight(
                final_disp, disp_pos,
                vmin, vmax,
                ylabel=disp_ylabel,
                xlabel=xlabel
            )
            plot_img_tight(
                diff, err_pos,
                0, (vmax - vmin)/2.0,
                ylabel=err_ylabel, xlabel=xlabel,
                cmap='gray'
            )

            method_idx += 1

        scene_idx += 1

    figure = plt.gcf()
    plt.tight_layout()
    figure.set_size_inches(12, 6.5)
    plt.subplots_adjust(wspace=0., hspace=0., left=0.02, right=1,
                        bottom=0.0125, top=1.)
    plt.show()

    # print(trim_list)
    # get last item in the folder...


def prog_plot():
    """
    Plots a scene with each method at a series of stages...

    Returns
    -------

    """
    scene = 'vinyl/'
    out_dir = 'output/additional/' + scene

    gt_name = 'data/additional/' + scene + 'gt_disp_lowres.pfm'
    gt, _ = fileIO.readPFM(gt_name)

    vmin = np.amin(gt)
    vmax = np.amax(gt)

    methods = {
        'old': 'traditional (a)',
        # 'inter_irls': 'Warp Free (b)',
        'gc_model': 'GC Model (d)'
    }
    method_alphas = {
        'old': 0.5,
        # 'inter_irls': 0.001,
        'gc_model': 2.0
    }

    solve_lists = {
        'old': ['0_0', '0_1', '2_1', '2_99'],
        # 'inter_irls': ['0_4', '0_9', '0_24', '2_49'],
        'gc_model': ['0_0', '0_1', '0_4', '0_99']
    }

    solve_titles = ['1st Solve', '2nd Solve', '5th Solve', 'Final Result']

    method_idx = 0
    n_methods = len(methods.keys())
    for method in methods.keys():
        print(method)
        folder = out_dir + method + '/alpha=' + str(method_alphas[method]) + '/'
        solve_list = solve_lists[method]
        solve_idx = 0
        n_solves = len(solve_list)
        for solve in solve_list:
            fname = 'Disparity_' + solve + '.npy'
            path = folder + fname
            print(path)
            field = np.load(path)
            field_pos = (2 * n_methods, n_solves,
                         2 * method_idx * n_solves + solve_idx + 1)

            if solve_idx == 0:
                field_ylabel = methods[method]
                err_ylabel = methods[method] + ' error'
            else:
                field_ylabel = None
                err_ylabel = None

            if method_idx == 1:
                err_xlabel = solve_titles[solve_idx]
            else:
                err_xlabel = None

            err = np.abs(gt - field)
            err_pos = (2 * n_methods, n_solves,
                       2 * method_idx * n_solves + solve_idx + n_solves + 1)

            plot_img_tight(field, field_pos, vmin, vmax, ylabel=field_ylabel
            )

            plot_img_tight(err, err_pos, vmin=0, vmax=1, ylabel=err_ylabel,
                           xlabel=err_xlabel)
            solve_idx += 1

        method_idx += 1
    plt.rc('font', size=15)
    figure = plt.gcf()
    plt.tight_layout()
    figure.set_size_inches(10, 10)
    plt.subplots_adjust(wspace=0., hspace=0., left=0.02, right=1,
                        bottom=0.02, top=1.)
    plt.show()


# zoom_plot()
plot_n_scenes()
# prog_plot()
