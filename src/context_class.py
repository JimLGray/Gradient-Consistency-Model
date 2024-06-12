# Module context_class.py

import numpy as np
import copy



class Context:
    """
    Contains all the information the algorithm needs to run.
    """
    __slots__ = ('filenames', 'baselines', 'ref_baseline', 'decimate_ratio',
                 'scale', 'focal_length', 'dH', 'delta_x', 'gt_disparity',
                 'gt_depth', 'w_init', 'data_cost_func', 'reg_cost_func',
                 'grad_err', 'ctf', 'lpf_err', 'w_err_method',
                 'filter_grad_err', 'valid_check', 'uniform_weights', 'vmin',
                 'vmax', 'reg_sigma', 'data_sigma', 'reg_consts',
                 'current_reg_const', 'acc_consts', 'current_acc_const',
                 'dynamic_d_sig', 'normalise_opposites', 'normalise_views',
                 'boundary_ext', 'grad_b_ext', 'tol',
                 'n_levels', 'n_warps_list', 'total_max_warps',
                 'downsample_ctf', 'downsample_multires', 'multi_res_levels',
                 'apply_limit', 'irls_stages', 'max_iter', 'noise_sigma',
                 'add_noise', 'noise_seed_ref', 'noise_seed_tgt', 'est_noise',
                 'noise_epsilon', 'start_n',
                 'increment', 'power', 'stages', 'res_dir', 'res_filename',
                 'save_plots', 'interpolate_in_IRLS', 'stability_check',
                 'prog_inc', 'median_filter', 'raw_imgs', 'quadratic_acc',
                 'four_neighbours', 'successive_filtering', 'warp_mode',
                 'warp_boundary', 'outside_image', 'grad_scaling',
                 'view_scaling', 'warp_top_level', 'save_weights',
                 'baseline_normalisation', 'snr_weights', 'reg_normalisation',
                 'confidence_metric', 'coarse_err', 'gc_err', 'schedule_stages',
                 'delta_w_vpq', 'baseline_noise', 'radial_distortion')
    def __init__(self):
        # Image and view information.
        self.filenames = []
        self.baselines = None
        self.ref_baseline = np.array([0, 0])
        self.baseline_normalisation = False
        self.decimate_ratio = 1
        self.scale = 1  # Distance between cameras.
        self.focal_length = 1
        self.dH = 1
        self.delta_x = 0
        self.baseline_noise = 0.0
        self.radial_distortion = 0.0

        # Starting Disparity information
        self.gt_disparity = None
        self.gt_depth = None
        self.w_init = None

        # Algorithm settings
        self.data_cost_func = 'L2'
        # {'L2', 'L1', 'L2-L1', 'WL-1', 'Welsch'} are valid choices
        self.reg_cost_func = 'L2'
        # accumulate in the quadratic domain?
        self.quadratic_acc = False
        # 4 neighbours or 8 neighbours for the regulariser
        self.four_neighbours = False
        self.grad_err = False
        self.ctf = False
        self.lpf_err = False
        self.w_err_method = 'regulariser'
        # 'regulariser', 'dw', 'gaussian_variance', 'max'

        # Weights Settings
        self.filter_grad_err = False
        self.valid_check = False
        self.uniform_weights = True
        self.vmin = -np.inf
        self.vmax = np.inf
        self.median_filter = False
        self.raw_imgs = False
        self.grad_scaling = False
        self.view_scaling = True
        self.apply_limit = False
        self.snr_weights = False
        self.reg_normalisation = False
        self.confidence_metric = False
        self.coarse_err = True
        self.gc_err = True
        self.schedule_stages = 5
        self.delta_w_vpq = False

        self.reg_sigma = 1
        self.data_sigma = 1  # this is dynamically set. Just bear that in mind.
        self.reg_consts = [1]
        self.current_reg_const = 1
        self.acc_consts = [1]
        self.current_acc_const = 1
        self.dynamic_d_sig = False
        self.normalise_opposites = False
        self.normalise_views = False
        self.boundary_ext = 'edge'
        self.grad_b_ext = 'edge'
        self.tol = 1e-3

        # Coarse to Fine Settings
        self.n_levels = 0
        self.n_warps_list = [0, 0]
        self.warp_top_level = False
        self.total_max_warps = 30
        self.downsample_ctf = False
        self.downsample_multires = False
        # This is about whether we downsample every multi_res_levels.
        self.successive_filtering = False
        # This is about whether at each scale, we simply filter the
        # finer scale, or filter the maximum scale.

        # Multires settings
        self.multi_res_levels = 1
        # set to negative value to not limit flows.

        # Warping settings...
        self.warp_mode = 'cubic'
        self.warp_boundary = 'edge'
        self.outside_image = False
        # if this is set to true, if data is warped in from outside the image,
        # we just use the reference image.

        # IRLS and CG settings
        self.irls_stages = 1
        self.interpolate_in_IRLS = True
        self.max_iter = 0
        self.stability_check = True

        # noise settings
        self.noise_sigma = 0.01
        self.add_noise = False
        self.noise_seed_ref = 0
        self.noise_seed_tgt = 1
        self.est_noise = False
        self.noise_epsilon = 1e-6

        # Progressive inclusion of views settings
        self.prog_inc = False
        self.start_n = 2
        self.increment = 1
        self.power = 1
        self.stages = 1

        # Output Settings
        self.res_dir = ''
        self.res_filename = ''
        self.save_plots = False
        self.save_weights = False

    def copy(self):
        """
        Does a basic copy
        Returns
        -------

        """
        new_params = copy.deepcopy(self)
        return new_params

    def max_warps(self):
        """
        Calculates the maximum number of warps possible for this context.
        Returns
        -------

        """
        return self.stages * sum(self.n_warps_list)

    def to_dict(self):
        """
        Returns a dictionary version of this object. The key application is so
        we can save the object as a .json file, so we don't include the numpy
        arrays.

        Returns
        -------

        """
        d = dict()
        for field in self.__slots__:
            obj = self.__getattribute__(field)
            if field == 'baselines' or field == 'ref_baseline':
                d[field] = obj.tolist()
            elif type(obj) == np.ndarray:
                d[field] = None
            else:
                d[field] = obj
        return d


class State:
    """
    Contains information on which warp you are at and which level you are at in
    the coarse to fine framework. Also will contain which stage you are at in
    the IRLS scheme
    """
    __slots__ = ['prog_inc_stage', 'ctf_level', 'total_ctf_levels',
                 'warp_stage', 'total_warp_stages', 'irls_stage',
                 'total_irls_stages', 'cur_baselines', 'start', 'stop',
                 'disp_lim', 'vmin', 'vmax', 'current_d_sig']
    def __init__(self, start=0, stop=1):
        # which progressive inclusion of views we are at
        self.prog_inc_stage = 0
        # coarse to fine level. Lower is coarser
        self.ctf_level = 0
        self.total_ctf_levels = 0
        # which warp stage we are at
        self.warp_stage = 0
        self.total_warp_stages = 0
        # which IRLS stage we are at
        self.irls_stage = 0
        self.total_irls_stages = 0
        self.cur_baselines = np.array([[0, 0]])
        self.start = start
        self.stop = stop
        self.disp_lim = 1
        self.vmin = -np.inf
        self.vmax = np.inf
        self.current_d_sig = 1.0

    def __str__(self):
        s = 'Progressive Inclusion of Views Stage: ' + \
            str(self.prog_inc_stage) + '\n' + \
            'Coarse to Fine Level: ' + str(self.ctf_level) + '\n' + \
            'Warp Stage: ' + str(self.warp_stage) + '\n' + \
            'IRLS Stage: ' + str(self.irls_stage) + '\n' + \
            'Start Res: ' + str(self.start) + '\n' + \
            'Stop Res:' + str(self.stop)
        return s

    def one_liner(self, irls=False):
        """
        Writes out the information in one line
        Returns
        -------

        """
        s = 'prog=' + str(self.prog_inc_stage) + \
            '_ctf=' + str(self.ctf_level) + \
            '_warp=' + str(self.warp_stage)
        if irls:
            s = s + '_irls=' + str(self.irls_stage)

        return s

    def is_first_stage(self):
        """
        Checks if we are in the first stage of the algorithm.

        Returns
        -------
        bool
        """
        if self.prog_inc_stage != 0:
            return False
        elif self.total_irls_stages != 0:
            return False
        elif self.total_warp_stages != 0:
            return False
        elif self.total_ctf_levels != 0:
            return False
        else:
            return True

    def update_ctf_level(self, n: int):
        """
        Sets ctf_level to n and then increases the total_ctf_levels by the
        appropriate amount
        Parameters
        ----------
        n: int
            integer to increase ctf level to

        Returns
        -------
        """
        self.ctf_level = n
        if n != 0 or not self.is_first_stage():
            self.total_ctf_levels += 1

    def update_warp_stage(self, n: int):
        """
        Sets warp_stage to n and then increases total_warp_stages by the right
        amount
        Parameters
        ----------
        n:int
            integer to increase warp_stage to

        """
        self.warp_stage = n
        if n != 0 or not self.is_first_stage():
            self.total_warp_stages += 1

    def update_irls_stage(self, n: int):
        """
        Sets irls stage to n and then increase irls total stages by the right
        amount
        Parameters
        ----------
        n: int
            integer to increase irls_stage to

        """
        self.irls_stage = n
        if n != 0 or not self.is_first_stage():
            self.total_irls_stages += 1

    def current_n_views(self, context: Context):
        """
        Calculates the current number of views using the context. Uses equation
        (n_0 + n) ^ power

        Parameters
        ----------
        context: Context
            context object. Contains algorithm parameters. The number of views
            at the start and the increment and power are key here.

        Returns
        -------
        n_views: int

        Notes
        -----
        n_0 is the number of views at the start.
        n is calculated by multiplying the current progressive inclusion of
        views stage by the increment.
        """
        n_0 = context.start_n
        n = self.prog_inc_stage * context.increment
        return (n_0 + n) ** context.power

    def calculate_sig_d(self, grads: np.ndarray, diff: np.ndarray,
                        w: np.ndarray):
        """
        RMSE of photometric error calculated the easy way. Done only on the
        views adjacent to the reference view.

        Parameters
        ----------
        grads
        diff
        w

        Returns
        -------

        """
        n_targets = grads.shape[0]
        n_res = grads.shape[1]
        height = grads.shape[2]
        width = grads.shape[3]

        n_close = 0
        s_err = 0

        for p in range(n_targets):
            B = self.cur_baselines[p + 1]
            if np.max(np.abs(B)) > 1.0:
                # If the uniform norm is bigger than 1
                continue
            for q in range(n_res):
                err = np.nansum((grads[p, q] * w + diff[p, q])**2)
                s_err += err

            n_close += 1

        d_sig = np.sqrt(s_err / (height * width * n_res * n_close))
        if d_sig < self.current_d_sig and d_sig > 1e-2:
            self.current_d_sig = d_sig
        elif d_sig <= 1e-2:
            self.current_d_sig = 1e-2

    def update_limit(self):
        """
        Set the limit to be 2^q / |max_Bp|

        Parameters
        ----------

        Returns
        -------

        """
        max_Bp = self.cur_baselines[-1]
        abs_Bp = np.sqrt(max_Bp[0] ** 2 + max_Bp[1]**2)
        q = self.stop - 1
        self.disp_lim = 2 ** q / abs_Bp


    # def update_w(self, w: np.ndarray):
    #     """
    #     sets self.w and moves the old value to the self.prev_w
    #     Parameters
    #     ----------
    #     w
    #
    #     Returns
    #     -------
    #
    #     """
    #     self.prev_w = np.copy(self.w)
    #     self.w = np.copy(w)


def make_paramset(input_path, output_path, method, alpha, schedule_solves):
    import regex
    from data_io import fileIO

    if 'middlebury_2006' in input_path:
        img_names = [
            'view1.png',
            'view0.png',
            'view2.png',
            'view3.png',
            'view4.png',
            'view5.png',
            'view6.png'
        ]
        sorted_names = []

        for name in img_names:
            sorted_names.append(input_path + name)
        sorted_baselines = [
            [0, 0],
            [0, -1],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
        ]
        trad_levels = 6
        gc_multi_res = 6
        trad_max_warps = 6
        hybrid_multi_res = 3
        hybrid_levels = 2
        hybrid_max_warps = 4
        vmin = -2 ** (gc_multi_res - 1)
        vmax = 2 ** (gc_multi_res - 1)
        gt_name = input_path + 'disp1.png'

    else:
        img_names = [
            'input_Cam040.png', 'input_Cam004.png', 'input_Cam013.png',
            'input_Cam022.png', 'input_Cam031.png', 'input_Cam036.png',
            'input_Cam037.png', 'input_Cam038.png', 'input_Cam039.png',
            'input_Cam041.png', 'input_Cam042.png', 'input_Cam043.png',
            'input_Cam044.png', 'input_Cam049.png', 'input_Cam058.png',
            'input_Cam067.png', 'input_Cam076.png'
        ]
        f_names = []

        for name in img_names:
            f_names.append(input_path + name)

        baselines, valid_names = regex.regex_light_field(f_names)

        sorted_baselines, sorted_names = regex.sort_baselines_names(baselines,
                                                                    valid_names)
        trad_levels = 3
        trad_max_warps = 3
        gc_multi_res = 3
        # hybrid doesn't make sense here so we just use gc parameters here
        hybrid_multi_res = gc_multi_res
        hybrid_levels = 1
        hybrid_max_warps = 1
        vmin, vmax = fileIO.read_vmin_vmax(input_path + 'parameters.cfg')
        gt_name = input_path + 'gt_disp_lowres.pfm'

    gc_warps = 1
    trad_multi_res = 1
    gc_levels = 1

    start_n = 5
    stages = 4
    increment = 4

    if method == 'old':
        n_levels = trad_levels
        multi_res_levels = trad_multi_res
        total_max_warps = trad_max_warps
        outside_image = True
        gc_err = False
        delta_w_vpq = False
        coarse_err = False
        grad_err = False
        normalise_opposites = False
    elif method == 'naive':
        n_levels = 1
        multi_res_levels = 1
        total_max_warps = 1
        outside_image = True
        gc_err = False
        delta_w_vpq = False
        coarse_err = False
        grad_err = False
        normalise_opposites = False
    elif method == 'prog_inc':
        n_levels = 1
        multi_res_levels = 1
        total_max_warps = stages
        outside_image = True
        grad_err = False
        delta_w_vpq = False
        gc_err = False
        coarse_err = False
        normalise_opposites = False
    elif method == 'gc_model':
        n_levels = gc_levels
        multi_res_levels = gc_multi_res
        total_max_warps = gc_warps
        outside_image = True
        delta_w_vpq = True
        grad_err = True
        gc_err = True
        coarse_err = True
        normalise_opposites = True
    elif method == 'hybrid':
        n_levels = hybrid_levels
        multi_res_levels = hybrid_multi_res
        total_max_warps = hybrid_max_warps
        outside_image = True
        delta_w_vpq = True
        grad_err = True
        gc_err = True
        coarse_err = True
        normalise_opposites = True
    elif method == 'gc_no_ge':
        n_levels = gc_levels
        multi_res_levels = gc_multi_res
        total_max_warps = gc_warps
        outside_image = True
        delta_w_vpq = True
        grad_err = True
        grad_err = True
        gc_err = False
        coarse_err = True
        normalise_opposites = True
    elif method == 'gc_no_cse':
        n_levels = gc_levels
        multi_res_levels = gc_multi_res
        total_max_warps = gc_warps
        outside_image = True
        delta_w_vpq = True
        grad_err = True
        grad_err = True
        gc_err = True
        coarse_err = False
        normalise_opposites = True
    else:
        raise ValueError('Invalid param_set')


    parameters = {
        'filenames': sorted_names,
        'baselines': sorted_baselines,
        'data_sigma': 0.01,
        'reg_sigma': 1.0,
        'dynamic_d_sig': True,
        'data_cost_func': 'L1',
        'reg_cost_func': 'L1',
        'max_iter': 512 * 512,

        'irls_stages': 100,
        'schedule_stages': schedule_solves,
        'n_warps_list': [1, 1, 1, 1, 1, 1],
        'current_reg_const': alpha,
        'interpolate_in_IRLS': True,
        'n_levels': n_levels,
        'multi_res_levels': multi_res_levels,
        'total_max_warps': total_max_warps,

        'median_filter': True,
        'reg_normalisation': True,
        'four_neighbours': True,
        'warp_top_level': True,
        'warp_boundary': 'edge',
        'warp_mode': 'cubic',
        'apply_limit': True,
        'stability_check': False,
        'prog_inc': False,
        'grad_scaling': False,
        'view_scaling': False,

        'normalise_opposites': normalise_opposites,
        'grad_err': grad_err,
        'gc_err': gc_err,
        'coarse_err': coarse_err,
        'delta_w_vpq': delta_w_vpq,

        'save_weights': False,
        'snr_weights': False,
        'confidence_metric': False,
        'normalise_views': False,
        'raw_imgs': True,
        'outside_image': outside_image,

        'noise_epsilon': 1e-4,
        'vmin': vmin,
        'vmax': vmax,
        'gt_name': gt_name,
        'res_dir': output_path,

        'start_n': start_n,
        'stages': stages,
        'increment': increment,
        'baseline_noise': 0,
        'radial_distortion': 0,
    }
    return parameters
