# Module img.py

"""
Images class which stores a series of images as a collection of spline
coefficients in numpy arrays.
"""

import splines as sp
import numpy as np
from data_io import fileIO, plot
from context_class import Context, State
from copy import deepcopy
from scipy import ndimage, io, signal
from simple import valid_data, gaussian_sq_filter2d, dog_sq_filter1d, get_sigma, \
    derivative1d
import warp
from regex import sort_baselines_names


class Images:
    """
    Stores the images as a series of spline coefficients. Uses numpy arrays to
    do this.

    Attributes
    ----------
    baselines: tuple | np.ndarray
        the baselines of all the images. ref should have (0,0) (y, x) style.
    n_imgs: int
        number of images.
    n_targets: int
        number of target images
    spl_coeffs: np.ndarray
        spline coefficients as calculated by the functions in splines.
        Dimensions of the form of
        [image, res, height, width]
    height: int
        height of the images
    width: int
        width of the images.
    ref_stack: np.ndarray
        stack of reference images at different scales
    reg_g_stack: np.ndarray
        stack of gradients images of ref at different scales.


    """
    __slots__ = ('baselines', 'n_imgs', 'n_targets', 'n_res', 'outside_image',
                 'height', 'width', 'spl_coeffs', 'noise_epsilon',
                 'baseline_normalisation', 'ref_stack', 'ref_g_stack',
                 'v_noise_stack', 'gv_noise_stack', 'max_height', 'max_width')

    def __init__(self, context: Context = None, state=None,
                 spl_coeffs: np.ndarray = None,
                 baselines: np.ndarray = None):
        """
        Opens up the images and calculates the spline coefficients and stores
        them.

        Parameters
        ----------
        context: Context
            context object containing dimensions and filenames.
        """
        if context is not None and state is not None:
            self.baselines = deepcopy(context.baselines)
            self.baselines[:, 1] = self.baselines[:, 1] * -1
            # Must reverse x coordinate
            self.n_imgs = len(self.baselines)
            if len(context.filenames) != self.n_imgs:
                raise ValueError('Mismatch between number of filenames and '
                                 'baselines')
            self.n_targets = self.n_imgs - 1
            self.n_res = state.stop - state.start
            self.outside_image = context.outside_image
            ref = fileIO.open_img(context.filenames[0])
            self.max_height = ref.shape[0]
            self.max_width = ref.shape[1]
            self.spl_coeffs = np.empty((self.n_imgs, self.n_res,
                                        self.max_height, self.max_width,))
            self.noise_epsilon = context.noise_epsilon
            self.baseline_normalisation = context.baseline_normalisation
            for p in range(self.n_imgs):
                idx = 0
                img = fileIO.open_img(context.filenames[p])
                for q in range(state.start, state.stop):
                    sigma = get_sigma(q)
                    f_img = ndimage.gaussian_filter(img, sigma, mode='nearest')
                    self.spl_coeffs[p, idx] = sp.spline_coeffs(f_img)
                    idx += 1

        if spl_coeffs is not None and baselines is not None:
            self.baseline_normalisation = False
            self.baselines = np.copy(baselines)
            self.noise_epsilon = 1e-4
            self.spl_coeffs = np.copy(spl_coeffs)
            self.n_imgs = spl_coeffs.shape[0]
            self.n_res = spl_coeffs.shape[1]
            self.max_height = spl_coeffs.shape[2]
            self.max_width = spl_coeffs.shape[3]
            if self.n_imgs != baselines.shape[0]:
                raise ValueError('Mismatch between number of filenames and '
                                 'baselines')
            self.baselines[:, 1] = self.baselines[:, 1] * -1

        v_0 = np.zeros((self.max_height, self.max_width, 2))
        self.ref_stack = np.empty(
            (self.n_res, self.max_height, self.max_width,))
        self.ref_g_stack = np.empty(
            (self.n_res, self.max_height, self.max_width, 2))
        self.v_noise_stack = np.zeros_like(self.ref_stack)
        self.gv_noise_stack = np.zeros_like(self.ref_g_stack)
        for q in range(self.n_res):
            if context is not None and state is not None:
                sigma = get_sigma(q + state.start)
            else:
                sigma = get_sigma(q)
            ref_coeffs = self.spl_coeffs[0, q]
            self.ref_stack[q] = sp.interpolate(ref_coeffs, v_0)
            self.ref_g_stack[q, :, :, 0] = sp.interpolate(ref_coeffs, v_0,
                                                          order=(1, 0))
            self.ref_g_stack[q, :, :, 1] = sp.interpolate(ref_coeffs, v_0,
                                                          order=(0, 1))

            v_noise_q = gaussian_sq_filter2d(self.v_noise_stack[q], sigma,
                                             mode='constant', cval=1. / 6.)
            self.v_noise_stack[q] = sp.spline_coeffs(v_noise_q)
            gv_noise_n = dog_sq_filter1d(self.gv_noise_stack[q, :, :, 0], 0,
                                         sigma, mode='constant', cval=1. / 12.)
            gv_noise_m = dog_sq_filter1d(self.gv_noise_stack[q, :, :, 1], 1,
                                         sigma, mode='constant', cval=1. / 12.)
            self.gv_noise_stack[q, :, :, 0] = sp.spline_coeffs(gv_noise_n)
            self.gv_noise_stack[q, :, :, 1] = sp.spline_coeffs(gv_noise_m)

    @classmethod
    def open(cls, context: Context, state: State):
        """
        Opens the images and calculates the spline coefficients and stores them.

        Parameters
        ----------
        context: Context
            context object containing dimensions and filenames.

        Returns
        -------
        An instance of the Images class.
        """
        return cls(context=context, state=state)

    @classmethod
    def set(cls, spl_coeffs: np.ndarray, baselines: np.ndarray):
        """
        Sets the items directly using info from spl_coeffs etc.
        Parameters
        ----------
        spl_coeffs
        baselines

        Returns
        -------

        """
        return cls(spl_coeffs=spl_coeffs, baselines=baselines)

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Gets the gradient vector of the pth image at the qth scale

        The relevant symbol is nabla I_pq.

        Parameters
        ----------
        p: int
            the index of the image.
        q: int
            scale of the image in question
        w: np.ndarray
            The disparity used to interpolate the image.

        Returns
        -------
        nabla_Ipq: np.ndarray
            the gradient vector along one or both axes.
        """
        # This is different because we don't downscale the
        # splines in this implementation.
        if p == 0:
            return self.ref_g_stack[q]
        else:
            v = np.empty((self.max_height, self.max_width, 2))
            v[:, :, 0] = self.baselines[p, 0] * w
            v[:, :, 1] = self.baselines[p, 1] * w

            nabla_Ipq = np.empty((self.max_height, self.max_width, 2))
            c = self.spl_coeffs[p, q]

            nabla_Ipq_0 = sp.interpolate(c, v, order=(1, 0))
            nabla_Ipq_1 = sp.interpolate(c, v, order=(0, 1))

            if self.outside_image:
                f = np.empty_like(v)
                f[:, :, 0] = v[:, :, 1]
                f[:, :, 1] = v[:, :, 0]
                ref_0 = self.ref_g_stack[q, :, :, 0]
                ref_1 = self.ref_g_stack[q, :, :, 1]
                nabla_Ipq_0 = outside_image_option(nabla_Ipq_0, ref_0, f)
                nabla_Ipq_1 = outside_image_option(nabla_Ipq_1, ref_1, f)

            nabla_Ipq[:, :, 0] = nabla_Ipq_0
            nabla_Ipq[:, :, 1] = nabla_Ipq_1

            return nabla_Ipq

    def get_dot_grad(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Gets the gradient averaged between the ref view and the tgt view dotted
        with the baseline.

        The relevant symbol is g_pq

        Parameters
        ----------
        p: int
            the index of the tgt image.
        q: int
            scale of the image in question
        w: np.ndarray
            The disparity used to interpolate the image.


        Returns
        -------
        g_pq: np.ndarray
            the gradient averaged between the ref view and the tgt view dotted
            with the baseline

        """
        ref_g = self.get_grad(0, q, w, level)
        tgt_g = self.get_grad(p, q, w, level)
        g_vec = (ref_g + tgt_g) / 2.
        if self.baseline_normalisation:
            baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                   + self.baselines[p, 1] ** 2)
            unit_baseline = self.baselines[p] / baseline_mag
            g_pq = np.sum(g_vec * unit_baseline, axis=-1)
        else:
            g_pq = np.sum(g_vec * self.baselines[p], axis=-1)

        return g_pq

    def get_diff(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Gets the difference between the reference image and the pth target image

        ie. delta Ipq = I_pq - I_0q

        Parameters
        ----------
        p: int
            the index of the tgt image.
        q: int
            scale of the image in question
        w: np.ndarray
            The disparity used to interpolate the image.

        Returns
        -------
        delta_Ipq: np.ndarray
            I_pq - I_0q
        """
        v = np.empty((self.max_height, self.max_width, 2))
        v[:, :, 0] = self.baselines[p, 0] * w
        v[:, :, 1] = self.baselines[p, 1] * w
        if self.baseline_normalisation:
            baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                   + self.baselines[p, 1] ** 2)
        else:
            baseline_mag = 1.0

        i_0q = self.ref_stack[q]
        i_pq = sp.interpolate(self.spl_coeffs[p, q], v)
        if self.outside_image:
            f = np.empty_like(v)
            f[:, :, 0] = v[:, :, 1]
            f[:, :, 1] = v[:, :, 0]
            i_pq = outside_image_option(i_pq, i_0q, f)
        delta_Ipq = i_pq - i_0q
        return delta_Ipq / baseline_mag

    def apply_weights(self, weights: np.ndarray, w: np.ndarray):
        """
        Applies the weights to produce a weighted sum of the g_pq and delta_Ipq

        Parameters
        ----------
        weights: np.ndarray
            weights to apply to Images class.
        w: np.ndarray
            disparity field.

        Returns
        -------

        """
        w_n_tgts = weights.shape[0]
        w_n_res = weights.shape[1]
        w_height = weights.shape[2]
        w_width = weights.shape[3]
        if (w_height != self.max_height or w_width != self.max_width or
                w_n_res != self.n_res or w_n_tgts != self.n_targets):
            raise ValueError('Dimensions of weights mismatch dimensions of '
                             'Images object')
        sum_g_pq = np.zeros((self.max_height, self.max_width))
        sum_delta_Ipq = np.zeros((self.max_height, self.max_width))
        for p in range(1, self.n_imgs):
            # Exclude reference, because all these terms already have the ref.
            for q in range(self.n_res):
                weight_pq = weights[p - 1, q]
                g_pq = self.get_dot_grad(p, q, w)
                delta_Ipq = self.get_diff(p, q, w)
                sum_g_pq += g_pq ** 2 * weight_pq
                sum_delta_Ipq += -g_pq * delta_Ipq * weight_pq
        return sum_g_pq, sum_delta_Ipq

    def grad_con(self, p, q, w):
        """
        Calculates the gradient consistency for a given pair of images. This
        is essentially a noise term.

        E[Q_pq^2] = | nabla I_pq - nabla I_0q |^2 |B_p|^2 / 4

        Parameters
        ----------
        p: int
            the index of the tgt image.
        q: int
            scale of the image in question
        w: np.ndarray
            The disparity used to interpolate the image.

        Returns
        -------
        E[G_pq^2]: np.ndarray
            as per above equation.

        """
        nabla_Ipq = self.get_grad(p, q, w)
        nabla_I0q = self.get_grad(0, q, w)
        # To save space B_mag, G_pq and g_diff_mag have already been squared.
        B_p = self.baselines[p]
        B_mag = B_p[0] ** 2 + B_p[1] ** 2
        g_diff = nabla_Ipq - nabla_I0q
        g_diff_mag = g_diff[:, :, 0] ** 2 + g_diff[:, :, 1] ** 2
        G_pq = g_diff_mag * B_mag / 4
        return G_pq

    def coarse_scale_err(self, p: int, q: int, w: np.ndarray):
        """
        Calculates the coares scale error at a given scale. Uses the an unsharp
        mask filter (hpf_gaussian) to determine the local change in w, w_hat.
        Then we use the equation
        E[C_pq^2] = w_hat^2 * g_pq^2

        Parameters
        ----------
        p: int
            the index of the tgt image.
        q: int
            scale of the image in question
        w: np.ndarray
            The disparity used to interpolate the image.

        Returns
        -------
        E[C_pq^2]: np.ndarray
            as per the above equation
        """
        # To save space w_hat, g_pq and C_pq have already been squared.
        w_hat = hpf_gaussian(w, q)
        g_pq = self.get_dot_grad(p, q, w) ** 2
        C_pq = w_hat ** 2 * g_pq
        return C_pq

    def valid_regions(self, p: int, w: np.ndarray):
        """
        Returns the regions of the images which are valid.

        Parameters
        ----------
        p: int
        w: np.ndarray

        Returns
        -------

        """
        v = np.empty((self.max_height, self.max_width, 2))
        v[:, :, 0] = self.baselines[p, 0] * w
        v[:, :, 1] = self.baselines[p, 1] * w
        v_regions = valid_data(v)
        return v_regions

    def valid_noise(self, p: int, q: int, w: np.ndarray):
        """
        Returns the noise associated with being near invalid data.

        Parameters
        ----------
        p: int
        q: int
        w: np.ndarray

        Returns
        -------

        """
        v = np.empty((self.max_height, self.max_width, 2))
        v[:, :, 0] = self.baselines[p, 0] * w
        v[:, :, 1] = self.baselines[p, 1] * w

        v_noise_pq = sp.interpolate(self.v_noise_stack[q], v)
        v_noise_pq = np.where(v_noise_pq > 0., v_noise_pq, 0)

        gv_noise_pq_n = sp.interpolate(self.gv_noise_stack[q, :, :, 0], v)
        gv_noise_pq_n = np.where(gv_noise_pq_n > 0., gv_noise_pq_n, 0)

        gv_noise_pq_m = sp.interpolate(self.gv_noise_stack[q, :, :, 1], v)
        gv_noise_pq_m = np.where(gv_noise_pq_m > 0., gv_noise_pq_m, 0)

        gv_noise_pq = gv_noise_pq_n * self.baselines[p, 0] ** 2 + \
                      gv_noise_pq_m * self.baselines[p, 1] ** 2

        return v_noise_pq, gv_noise_pq

    def prediction_err(self, w: np.ndarray):
        """

        Returns
        -------

        """
        rmse = 0
        ref = self.ref_stack[0]
        for p in range(self.n_targets):
            tgt_coeffs = self.spl_coeffs[p + 1, 0]
            v = np.empty((self.max_height, self.max_width, 2))
            v[:, :, 0] = w * self.baselines[p + 1, 0]
            v[:, :, 1] = w * self.baselines[p + 1, 1]
            tgt = sp.interpolate(tgt_coeffs, v)
            rmse += np.nanmean(np.sqrt((tgt - ref) ** 2))

        return rmse / self.n_targets


class ResampledImgs(Images):
    """
    A version of the Images class which essentially subsamples the images at
    small scales to save memory
    """
    __slots__ = ('spl_coeffs_list', 'heights', 'widths')

    def __init__(self, context: Context, state: State):
        self.baselines = deepcopy(context.baselines)
        self.baselines[:, 1] = self.baselines[:, 1] * -1
        self.baseline_normalisation = context.baseline_normalisation
        # Must reverse x coordinate
        self.n_imgs = len(self.baselines)
        if len(context.filenames) != self.n_imgs:
            raise ValueError('Mismatch between number of filenames and '
                             'baselines')
        self.n_targets = self.n_imgs - 1
        self.n_res = state.stop - state.start
        ref = fileIO.open_img(context.filenames[0])
        self.spl_coeffs_list = [np.ndarray] * self.n_res
        self.heights = [int] * self.n_res
        self.widths = [int] * self.n_res
        self.noise_epsilon = context.noise_epsilon
        self.outside_image = context.outside_image
        idx = 0

        for q in range(state.start, state.stop):
            sigma = get_sigma(q)
            h = np.ceil(ref.shape[0] / (2 ** q))  # + ref.shape[0] % (2 ** q)
            w = np.ceil(ref.shape[1] / (2 ** q))  # + ref.shape[1] % (2 ** q)
            self.heights[idx] = int(h)
            self.widths[idx] = int(w)
            self.spl_coeffs_list[idx] = np.empty(
                (self.n_imgs, self.heights[idx],
                 self.widths[idx]))
            # This is now organised by scale first, to keep things tidy memory
            # wise
            for p in range(self.n_imgs):
                img = fileIO.open_img(context.filenames[p])
                f_img = ndimage.gaussian_filter(img, sigma, mode='nearest')
                dec_img = f_img[::2 ** q, ::2 ** q]
                self.spl_coeffs_list[idx][p] = sp.spline_coeffs(dec_img)

            idx += 1

        self.height = self.heights[0]
        self.width = self.widths[0]

        v_0 = np.zeros((self.height, self.width, 2))
        self.ref_stack = np.empty((self.n_res, self.height, self.width,))
        self.ref_g_stack = np.empty((self.n_res, self.height, self.width, 2))
        if context.valid_check:
            self.v_noise_stack = np.zeros_like(self.ref_stack)
            self.gv_noise_stack = np.zeros_like(self.ref_g_stack)
            for q in range(self.n_res):
                sigma = get_sigma(q)
                scale = 2 ** q
                v_noise_q = gaussian_sq_filter2d(self.v_noise_stack[q], sigma,
                                                 mode='constant', cval=1. / 6.)
                self.v_noise_stack[q] = sp.spline_coeffs(v_noise_q)
                gv_noise_n = dog_sq_filter1d(self.gv_noise_stack[q, :, :, 0], 0,
                                             sigma, mode='constant',
                                             cval=1. / 12.)
                gv_noise_m = dog_sq_filter1d(self.gv_noise_stack[q, :, :, 1], 1,
                                             sigma, mode='constant',
                                             cval=1. / 12.)
                self.gv_noise_stack[q, :, :, 0] = sp.spline_coeffs(gv_noise_n)
                self.gv_noise_stack[q, :, :, 1] = sp.spline_coeffs(gv_noise_m)
        else:
            self.v_noise_stack = 0
            self.gv_noise_stack = 0
        for q in range(self.n_res):
            sigma = get_sigma(q)
            scale = 2 ** q
            g_scale = 2 ** (q + state.start)
            ref_coeffs = self.spl_coeffs_list[q][0]
            inter_res = sp.upscale_interpolate2d(ref_coeffs, v_0, scale)
            self.ref_stack[q] = inter_res
            inter_res = sp.upscale_interpolate2d(
                ref_coeffs,
                v_0,
                scale,
                order=(1, 0)
            ) / g_scale
            self.ref_g_stack[q, :, :, 0] = inter_res
            inter_res = sp.upscale_interpolate2d(
                ref_coeffs,
                v_0,
                scale,
                order=(0, 1)
            ) / g_scale
            self.ref_g_stack[q, :, :, 1] = inter_res

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Interpolate up to the top scale..
        Parameters
        ----------
        p
        q: int
            assume q=0 is the top level here. Note that this is distinct to the
            scale or "level" because the scale is not always 0 in a ctf, but for
            q it is.
        w

        Returns
        -------

        """
        if p == 0:
            return self.ref_g_stack[q]

        w_height = w.shape[0]
        w_width = w.shape[1]
        c = self.spl_coeffs_list[q][p]
        # scale = 2 ** (q + level * self.n_res)
        if self.n_res == 1:
            scale = 2 ** q
            grad_scale = 2 ** (q + level * self.n_res)
        else:
            scale = 2 ** q
            grad_scale = 2 ** (level * self.n_res)

        # zero padding for v is okay here, because it shouldn't affect the
        # results in neighbouring pixels. We also will trim it back to size.
        v = np.zeros((self.heights[0], self.widths[0], 2))

        if w_height != v.shape[0] or w_width != v.shape[1]:
            v[0:w_height, 0:w_width, 0] = self.baselines[p, 0] * w / grad_scale
            v[0:w_height, 0:w_width, 1] = self.baselines[p, 1] * w / grad_scale
            # this bit doesn't make sense???
        else:
            v[:, :, 0] = self.baselines[p, 0] * w / grad_scale
            v[:, :, 1] = self.baselines[p, 1] * w / grad_scale

        nabla_Ipq = np.empty((self.heights[0], self.widths[0], 2))
        Ipq_n = sp.upscale_interpolate2d(c, v, scale, (1, 0)) / grad_scale
        Ipq_m = sp.upscale_interpolate2d(c, v, scale, (0, 1)) / grad_scale

        if self.outside_image:
            I_0q = self.ref_g_stack[q]
            I_0q_n = I_0q[:, :, 0]
            I_0q_m = I_0q[:, :, 1]
            f = np.empty_like(v)
            f[:, :, 0] = v[:, :, 1]
            f[:, :, 1] = v[:, :, 0]
            # print(scale)
            Ipq_n = outside_image_option(Ipq_n, I_0q_n, f)
            Ipq_m = outside_image_option(Ipq_m, I_0q_m, f)

        nabla_Ipq[:, :, 0] = Ipq_n[0:self.heights[0], 0:self.widths[0]]
        nabla_Ipq[:, :, 1] = Ipq_m[0:self.heights[0], 0:self.widths[0]]
        return nabla_Ipq

    def get_diff(self, p: int, q: int, w: np.ndarray, level=0):
        """

        Parameters
        ----------
        p
        q
        w

        Returns
        -------

        """
        w_height = w.shape[0]
        w_width = w.shape[1]
        c_pq = self.spl_coeffs_list[q][p]
        if self.baseline_normalisation:
            baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                   + self.baselines[p, 1] ** 2)
        else:
            baseline_mag = 1.0
        if self.n_res == 1:
            scale = 2 ** q
            d_scale = 2 ** (q + level * self.n_res)
        else:
            scale = 2 ** q
            d_scale = 2 ** (level * self.n_res)
        # zero padding for v is okay here, because it shouldn't affect the
        # results in neighbouring pixels. We also will trim it back to size.
        v = np.zeros((self.heights[0], self.widths[0], 2))

        if w_height != v.shape[0] or w_width != v.shape[1]:
            v[0:w_height, 0:w_width, 0] = self.baselines[p, 0] * w / d_scale
            v[0:w_height, 0:w_width, 1] = self.baselines[p, 1] * w / d_scale
        else:
            v[:, :, 0] = self.baselines[p, 0] * w / d_scale
            v[:, :, 1] = self.baselines[p, 1] * w / d_scale

        I_0q = self.ref_stack[q]
        I_pq = sp.upscale_interpolate2d(c_pq, v, scale)
        if self.outside_image:
            f = np.empty_like(v)
            f[:, :, 0] = v[:, :, 1]
            f[:, :, 1] = v[:, :, 0]
            I_pq = outside_image_option(I_pq, I_0q, f)
        delta_Ipq = (I_pq - I_0q) * scale / baseline_mag
        return delta_Ipq[0:self.heights[0], 0:self.widths[0]]

    def save_warped_imgs(self, res_dir: str, w: np.ndarray, idx: int, level=0):
        w_imgs = np.empty((self.n_imgs, self.n_res, self.height, self.width))
        w_height = w.shape[0]
        w_width = w.shape[1]

        for p in range(self.n_imgs):
            if self.baseline_normalisation:
                baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                       + self.baselines[p, 1] ** 2)
            else:
                baseline_mag = 1.0

            for q in range(self.n_res):
                if p == 0:
                    i_pq = self.ref_stack[q]
                else:
                    if self.n_res == 1:
                        scale = 2 ** q
                        d_scale = 2 ** (q + level * self.n_res)
                    else:
                        scale = 2 ** q
                        d_scale = 2 ** (level * self.n_res)

                    v = np.zeros((self.heights[0], self.widths[0], 2))

                    if w_height != v.shape[0] or w_width != v.shape[1]:
                        v[0:w_height, 0:w_width, 0] = self.baselines[
                                                          p, 0] * w / d_scale
                        v[0:w_height, 0:w_width, 1] = self.baselines[
                                                          p, 1] * w / d_scale
                    else:
                        v[:, :, 0] = self.baselines[p, 0] * w / d_scale
                        v[:, :, 1] = self.baselines[p, 1] * w / d_scale

                    c_pq = self.spl_coeffs_list[q][p]
                    i_pq = sp.upscale_interpolate2d(c_pq, v, scale)
                    if self.outside_image:
                        f = np.empty_like(v)
                        f[:, :, 0] = v[:, :, 1]
                        f[:, :, 1] = v[:, :, 0]
                        i_pq = outside_image_option(i_pq, self.ref_stack[q], f)
                w_imgs[p, q] = i_pq
        f_name = res_dir + 'warped_imgs_' + str(idx) + '.npy'
        np.save(f_name, w_imgs)


class RawImages(Images):
    """
    Class that does much of the same things as the images, but does warping
    instead. We don't use splines for this, things are calculated directly.
    We try to copy the paper at doi: 10.1109/ICME.2017.8019377
    """
    __slots__ = ('heights', 'widths', 'warp_mode', 'warp_boundary',
                 'top_height', 'top_width', 'imgs_list', 'warp_top_level',
                 'raw_imgs')

    def __init__(self, context: Context, state: State):

        self.n_res = state.stop - state.start
        self.heights = [int] * self.n_res
        self.widths = [int] * self.n_res

        self.warp_mode = context.warp_mode
        self.warp_boundary = context.warp_boundary
        self.outside_image = context.outside_image
        self.noise_epsilon = context.noise_epsilon

        self.baselines = deepcopy(context.baselines)
        self.baselines = self.baselines.astype(float)
        self.baselines[:, 1] = self.baselines[:, 1] * -1
        self.baseline_normalisation = context.baseline_normalisation

        self.n_imgs = len(self.baselines)
        self.n_targets = self.n_imgs - 1
        sig_0 = get_sigma(0)

        if context.baseline_noise > 0:
            np.random.seed(context.noise_seed_tgt + 10)
            for idx in range(1, self.n_targets + 1):
                noise = np.random.normal(0, context.baseline_noise, 2)
                self.baselines[idx] += noise

        tmp = fileIO.open_img(context.filenames[0])
        K_1 = context.radial_distortion
        if np.abs(K_1) > 0:
            ref = radial_distortion(tmp, K_1)
        else:
            ref = tmp

        self.top_height = ref.shape[0]
        self.top_width = ref.shape[1]

        self.imgs_list = None
        self.warp_top_level = True
        self.raw_imgs = np.empty((self.n_imgs, ref.shape[0],
                                  ref.shape[1]))
        self.raw_imgs[0] = ref
        for p in range(1, self.n_imgs):
            tmp = fileIO.open_img(context.filenames[p])
            if np.abs(K_1) > 0:
                self.raw_imgs[p] = radial_distortion(tmp, K_1)
            else:
                self.raw_imgs[p] = tmp

        idx = 0
        for q in range(state.start, state.stop):
            h = np.ceil(ref.shape[0] / (2 ** q))  # + ref.shape[0] % (2**q)
            w = np.ceil(ref.shape[1] / (2 ** q))  # + ref.shape[1] % (2**q)
            self.heights[idx] = int(h)
            self.widths[idx] = int(w)
            idx += 1

        self.height = self.heights[0]
        self.width = self.widths[0]

    def __warp__(self, p: int, w: np.ndarray):
        """
        Does the warp at the top resolution and then returns the warped view

        Parameters
        ----------
        p
        w

        Returns
        -------

        """
        raw_img = self.raw_imgs[p]
        if p == 0 or np.allclose(w, 0):
            return raw_img
        else:
            if w.shape[0] != self.top_height or w.shape[1] != self.top_width:
                w = resample_2d(w, self.top_height, self.top_width)

            v = np.empty((self.top_height, self.top_width, 2))
            v[:, :, 1] = self.baselines[p, 0] * w
            v[:, :, 0] = self.baselines[p, 1] * w

            cond = self.outside_image and \
                   (self.warp_mode == 'bilinear' or self.warp_mode == 'linear')
            if cond:
                ref_raw = self.raw_imgs[0]
                warp_img = warp.bilnear_warp(ref_raw, raw_img, v)
            else:
                warp_img = warp.fast_warp(raw_img, v, self.warp_mode,
                                          self.warp_boundary)
                if self.outside_image:
                    ref_raw = self.raw_imgs[0]
                    warp_img = outside_image_option(warp_img, ref_raw, v)
            return warp_img

    def get_warp(self, p: int, q: int, w: np.ndarray, level=0):
        scale = level
        sig = get_sigma(q + scale)
        warp_img = self.__warp__(p, w)

        w_filt = ndimage.gaussian_filter(warp_img, sig, mode='nearest')
        w_img = w_filt[::2 ** scale, ::2 ** scale]

        return w_img

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Interpolate up to the top scale.
        Need to do this with derivative of gaussians instead of difference of
        gaussians.
        Parameters
        ----------
        p
        q: int
            assume q=0 is the top level here. Note that this is distinct to the
            scale or "level" because the scale is not always 0 in a ctf, but for
            q it is.
        w

        Returns
        -------
        """
        x_axis = (0, 1)
        y_axis = (1, 0)
        scale = level
        sig = get_sigma(q + scale)

        warp_img = self.__warp__(p, w)

        y_grad = ndimage.gaussian_filter(
            warp_img, sig, order=y_axis, mode='nearest'
        )
        x_grad = ndimage.gaussian_filter(
            warp_img, sig, order=x_axis, mode='nearest'
        )

        nabla_I_pq = np.empty((self.heights[0], self.widths[0], 2))
        nabla_I_pq[:, :, 0] = y_grad[::2 ** scale, ::2 ** scale]
        nabla_I_pq[:, :, 1] = x_grad[::2 ** scale, ::2 ** scale]
        return nabla_I_pq

    def get_diff(self, p: int, q: int, w: np.ndarray, level=0):
        i_0q = self.get_warp(0, q, w, level)
        i_pq = self.get_warp(p, q, w, level)
        if self.baseline_normalisation:
            baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                   + self.baselines[p, 1] ** 2)
        else:
            baseline_mag = 1.0
        return (i_pq - i_0q) / baseline_mag

    def save_warped_imgs(self, res_dir: str, w: np.ndarray, idx: int, level=0):
        """
        Saves the warped images to a file in res_dir. Each view is ordered by
        self.baselines, which should appear in the context.json file.
        The order of the resolutions is top resolution is 0th.
        Parameters
        ----------
        res_dir
        w

        Returns
        -------

        """
        w_imgs = np.empty((self.n_imgs, self.n_res, self.height, self.width))
        for q in range(self.n_res):
            for p in range(self.n_imgs):
                i_pq = self.get_warp(p, q, w, level)
                w_imgs[p, q] = i_pq
        f_name = res_dir + 'warped_imgs_' + str(idx) + '.npy'
        np.save(f_name, w_imgs)


class OverSampledImgs(RawImages):
    """
    Very similar to RawImages but uses twice resolution to perform warping and
    calculate gradients.
    """

    def __init__(self, context: Context, state: State):
        self.n_res = state.stop - state.start

        self.warp_mode = context.warp_mode
        self.warp_boundary = context.warp_boundary
        self.outside_image = context.outside_image
        self.noise_epsilon = context.noise_epsilon

        self.baselines = deepcopy(context.baselines)
        self.baselines[:, 1] = self.baselines[:, 1] * -1
        self.baseline_normalisation = context.baseline_normalisation
        self.n_imgs = len(self.baselines)
        self.n_targets = self.n_imgs - 1

        ref = fileIO.open_img(context.filenames[0])
        self.og_height = ref.shape[0]
        self.og_width = ref.shape[1]

        self.top_height = self.og_height * 2
        self.top_width = self.og_width * 2
        self.heights = [int] * self.n_res
        self.widths = [int] * self.n_res

        self.ovr_imgs = np.empty((self.n_imgs, self.top_height, self.top_width))
        self.ovr_imgs[0] = resample_2d(ref, self.top_height, self.top_width)

        for p in range(1, self.n_imgs):
            img_p = fileIO.open_img(context.filenames[p])
            self.ovr_imgs[p] = resample_2d(img_p, self.top_height,
                                           self.top_width)
        idx = 0
        for q in range(state.start, state.stop):
            h = np.ceil(ref.shape[0] / (2 ** q))  # + ref.shape[0] % (2**q)
            w = np.ceil(ref.shape[1] / (2 ** q))  # + ref.shape[1] % (2**q)
            self.heights[idx] = int(h)
            self.widths[idx] = int(w)
            idx += 1
        self.height = self.heights[0]
        self.width = self.widths[0]

    def __ovr_warp__(self, ovr_img, p: int, w: np.ndarray):
        """
        Warps the oversampled image and returns a oversampled result

        Parameters
        ----------
        p: int
            the image in question
        w: np.ndarray
            disparity field

        Returns
        -------

        """
        if p == 0 or np.allclose(w, 0):
            return ovr_img
        else:
            ovr_w = resample_2d(w * 2, self.top_height, self.top_width)
            v = np.empty((self.top_height, self.top_width, 2))
            v[:, :, 1] = self.baselines[p, 0] * ovr_w
            v[:, :, 0] = self.baselines[p, 1] * ovr_w
            ovr_warp = warp.fast_warp(ovr_img, v, self.warp_mode,
                                      self.warp_boundary)
            if self.outside_image:
                ovr_warp = outside_image_option(ovr_warp, self.ovr_imgs[0], v)
            return ovr_warp

    def get_warp(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Downsamples the oversampled images that are stored in memory. Also
        performs warpign operations where required.

        Parameters
        ----------
        p
        q
        w
        level
        """
        scale = level + 1  # since we're operating at 2x res
        sigma = get_sigma(q + scale)
        sig_0 = get_sigma(0)
        # filter before warping
        ovr_img = self.ovr_imgs[p]
        ovr_img = ndimage.gaussian_filter(ovr_img, sig_0, mode='nearest')
        ovr_warp = self.__ovr_warp__(ovr_img, p, w)
        # filter after warping before downsampling.
        filt_img = ndimage.gaussian_filter(ovr_warp, sigma, mode='nearest')
        i_pq = filt_img[::2 ** scale, ::2 ** scale]
        return i_pq

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Calculates the gradient at 2x res and then downsamples the result.
        Performs the warp at 2x res if required.

        Parameters
        ----------
        p
        q
        w
        level

        Returns
        -------

        """
        x_axis = (0, 1)
        y_axis = (1, 0)
        scale = level + 1  # since we're operating at 2x res
        sigma = get_sigma(q + scale)
        sig_0 = get_sigma(0)

        ovr_img = self.ovr_imgs[p]
        # ovr_warp = self.__ovr_warp__(ovr_img, p, w)

        # nabla_Ipq_n = ndimage.gaussian_filter(ovr_warp, sigma, order=y_axis,
        #                                       mode='nearest')
        # nabla_Ipq_m = ndimage.gaussian_filter(ovr_warp, sigma, order=x_axis,
        #                                       mode='nearest')

        x_g_unwarped = ndimage.gaussian_filter(ovr_img, sig_0, order=x_axis,
                                               mode='nearest')
        y_g_unwarped = ndimage.gaussian_filter(ovr_img, sig_0, order=y_axis,
                                               mode='nearest')

        x_grad = self.__ovr_warp__(x_g_unwarped, p, w)
        y_grad = self.__ovr_warp__(y_g_unwarped, p, w)
        nabla_Ipq_n = ndimage.gaussian_filter(y_grad, sigma, mode='nearest')
        nabla_Ipq_m = ndimage.gaussian_filter(x_grad, sigma, mode='nearest')

        nabla_I_pq = np.empty((self.heights[0], self.widths[0], 2))
        nabla_I_pq[:, :, 0] = nabla_Ipq_n[::2 ** scale, ::2 ** scale]
        nabla_I_pq[:, :, 1] = nabla_Ipq_m[::2 ** scale, ::2 ** scale]

        return nabla_I_pq * 2


class TopScaleSplines(Images):
    """
    A version of the Images class which subsamples the results of the spline
    interpolation
    """

    def __init__(self, context, state):
        self.baselines = deepcopy(context.baselines)
        self.baselines[:, 1] = self.baselines[:, 1] * -1
        self.baseline_normalisation = context.baseline_normalisation
        # Must reverse x coordinate
        self.n_imgs = len(self.baselines)
        if len(context.filenames) != self.n_imgs:
            raise ValueError('Mismatch between number of filenames and '
                             'baselines')
        self.n_targets = self.n_imgs - 1
        self.n_res = state.stop - state.start
        ref = fileIO.open_img(context.filenames[0])
        self.max_height = ref.shape[0]
        self.max_width = ref.shape[1]
        self.spl_coeffs = np.empty((self.n_imgs, self.n_res,
                                    self.max_height + 2, self.max_width + 2))

        self.heights = [int] * self.n_res
        self.widths = [int] * self.n_res
        self.noise_epsilon = context.noise_epsilon
        self.outside_image = context.outside_image

        idx = 0
        self.scale = 2 ** state.start
        scale = self.scale

        self.height = int(np.ceil(self.max_height / scale))
        self.width = int(np.ceil(self.max_width / scale))

        for q in range(state.start, state.stop):
            sigma = get_sigma(q)
            for p in range(self.n_imgs):
                img = fileIO.open_img(context.filenames[p])
                f_img = ndimage.gaussian_filter(img, sigma, mode='nearest')
                pad_f_img = np.pad(f_img, 1, mode='edge')
                self.spl_coeffs[p, idx] = sp.spline_coeffs(pad_f_img)

            idx += 1

        v_0 = np.zeros((self.max_height, self.max_width, 2))
        pad_v0 = np.pad(v_0, 1, mode='edge')[:, :, 1:-1]
        self.ref_stack = np.empty((self.n_res, self.height, self.width))
        self.ref_g_stack = np.empty((self.n_res, self.height, self.width, 2))

        self.v_noise_stack = 0
        self.gv_noise_stack = 0

        idx = 0

        for q in range(state.start, state.stop):
            sigma = get_sigma(q)
            ref_coeffs = self.spl_coeffs[0, idx]
            inter_res = sp.upscale_interpolate2d(ref_coeffs, pad_v0, 1)
            trim_res = inter_res[1:-1, 1:-1]

            g_res_y = sp.upscale_interpolate2d(
                ref_coeffs,
                pad_v0,
                1,
                order=(1, 0)
            )[1:-1, 1:-1] * 2 ** q
            g_res_x = sp.upscale_interpolate2d(
                ref_coeffs,
                pad_v0,
                1,
                order=(0, 1)
            )[1:-1, 1:-1] * 2 ** q

            if self.max_height == self.height:
                self.ref_stack[idx] = trim_res
                self.ref_g_stack[idx, :, :, 0] = g_res_y
                self.ref_g_stack[idx, :, :, 1] = g_res_x
            else:
                self.ref_stack[idx] = trim_res[::scale, ::scale]
                self.ref_g_stack[idx, :, :, 0] = g_res_y[::scale, ::scale]
                self.ref_g_stack[idx, :, :, 1] = g_res_x[::scale, ::scale]

            idx += 1

    def get_warp(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Interpolate at the top scale.
        Parameters
        ----------
        p
        q: int
            assume q=0 is the top level here. Note that this is distinct to the
            scale or "level" because the scale is not always 0 in a ctf, but for
            q it is.
        w

        Returns
        -------
        """
        if p == 0:
            return self.ref_stack[q]

        w_height = w.shape[0]
        w_width = w.shape[1]

        scale = self.scale

        if scale != 1:
            w_rescale = resample_2d(w, self.max_height, self.max_width)
        else:
            w_rescale = w
        v = np.zeros((self.max_height, self.max_height, 2))

        v[:, :, 0] = self.baselines[p, 0] * w_rescale
        v[:, :, 1] = self.baselines[p, 1] * w_rescale

        pad_v = np.pad(v, 1, mode='edge')[:, :, 1:-1]
        c = self.spl_coeffs[p, q]

        if scale == 1:
            i_pq = sp.upscale_interpolate2d(c, pad_v, 1)[1:-1, 1:-1]
        else:
            tmp = sp.upscale_interpolate2d(c, pad_v, 1)[1:-1, 1:-1]
            i_pq = tmp[::scale, ::scale]

        if self.outside_image:
            f = np.empty((self.height, self.width, 2))
            f[:, :, 0] = w * self.baselines[p, 1]
            f[:, :, 1] = w * self.baselines[p, 0]
            i_pq = outside_image_option(i_pq, self.ref_stack[q], f)
        return i_pq

    def get_diff(self, p: int, q: int, w: np.ndarray, level=0):
        i_0q = self.get_warp(0, q, w)
        i_pq = self.get_warp(p, q, w)
        if self.baseline_normalisation:
            baseline_mag = np.sqrt(self.baselines[p, 0] ** 2
                                   + self.baselines[p, 1] ** 2)
        else:
            baseline_mag = 1.0
        return (i_pq - i_0q) * 2 ** q / baseline_mag

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        if p == 0:
            return self.ref_g_stack[q]

        scale = self.scale

        w_height = w.shape[0]
        w_width = w.shape[1]

        if scale != 1:
            w_rescale = resample_2d(w, self.max_height, self.max_width)
        else:
            w_rescale = w
        v = np.zeros((self.max_height, self.max_height, 2))
        v[:, :, 0] = self.baselines[p, 0] * w_rescale
        v[:, :, 1] = self.baselines[p, 1] * w_rescale

        pad_v = np.pad(v, 1, mode='edge')[:, :, 1:-1]
        c = self.spl_coeffs[p, q]

        if scale == 1:
            Ipq_n = sp.upscale_interpolate2d(c, pad_v, 1, (1, 0))[1:-1, 1:-1]
            Ipq_m = sp.upscale_interpolate2d(c, pad_v, 1, (0, 1))[1:-1, 1:-1]
        else:
            g_res_y = sp.upscale_interpolate2d(c, pad_v, 1, (1, 0))[1:-1, 1:-1]
            g_res_x = sp.upscale_interpolate2d(c, pad_v, 1, (0, 1))[1:-1, 1:-1]
            Ipq_n = g_res_y[::scale, ::scale]
            Ipq_m = g_res_x[::scale, ::scale]

        if self.outside_image:
            I_0q = self.ref_g_stack[q]
            I_0q_n = I_0q[:, :, 0]
            I_0q_m = I_0q[:, :, 1]
            f = np.empty((self.height, self.width, 2))
            f[:, :, 0] = w * self.baselines[p, 1]
            f[:, :, 1] = w * self.baselines[p, 0]
            # print(scale)
            Ipq_n = outside_image_option(Ipq_n, I_0q_n, f)
            Ipq_m = outside_image_option(Ipq_m, I_0q_m, f)

        nabla_Ipq = np.empty((self.height, self.width, 2))
        nabla_Ipq[:, :, 0] = Ipq_n * 2 ** q
        nabla_Ipq[:, :, 1] = Ipq_m * 2 ** q
        return nabla_Ipq

    def save_warped_imgs(self, res_dir: str, w: np.ndarray, idx: int, level=0):
        w_imgs = np.empty((self.n_imgs, self.n_res, self.height, self.width))
        for q in range(self.n_res):
            for p in range(self.n_imgs):
                i_pq = self.get_warp(p, q, w, level)
                w_imgs[p, q] = i_pq
        f_name = res_dir + 'warped_imgs_' + str(idx) + '.npy'
        np.save(f_name, w_imgs)


class GaussianSplines(TopScaleSplines):
    """
    Like TopScaleSplines but we combine the interpolation and the Gaussian
    Filtering stage. This means that we only do downsampling at the end.
    """

    def __init__(self, context, state):
        self.baselines = deepcopy(context.baselines)
        self.baselines[:, 1] = self.baselines[:, 1] * -1
        self.baseline_normalisation = context.baseline_normalisation
        # Must reverse x coordinate
        self.n_imgs = len(self.baselines)
        if len(context.filenames) != self.n_imgs:
            raise ValueError('Mismatch between number of filenames and '
                             'baselines')
        self.n_targets = self.n_imgs - 1
        self.n_res = state.stop - state.start
        ref = fileIO.open_img(context.filenames[0])
        self.max_height = ref.shape[0]
        self.max_width = ref.shape[1]

        pad_width = 2 ** (self.n_res + 1)
        self.pad_width = pad_width

        self.spl_coeffs = np.empty((self.n_imgs,
                                    self.max_height + 2 * pad_width,
                                    self.max_width + 2 * pad_width))

        self.heights = [int] * self.n_res
        self.widths = [int] * self.n_res
        self.noise_epsilon = context.noise_epsilon
        self.outside_image = context.outside_image

        idx = 0
        self.scale = 2 ** state.start
        scale = self.scale

        self.height = int(np.ceil(self.max_height / scale))
        self.width = int(np.ceil(self.max_width / scale))

        for p in range(self.n_imgs):
            img = fileIO.open_img(context.filenames[p])
            pad_img = np.pad(img, pad_width, mode='edge')
            self.spl_coeffs[p] = sp.spline_coeffs(pad_img)

        v_0 = np.zeros((self.max_height, self.max_width, 2))
        pad_v0 = np.pad(v_0, pad_width, mode='edge')[:, :, pad_width:-pad_width]
        self.ref_stack = np.empty((self.n_res, self.height, self.width))
        self.ref_g_stack = np.empty((self.n_res, self.height, self.width, 2))

        self.v_noise_stack = 0
        self.gv_noise_stack = 0

        for q in range(state.start, state.stop):
            ref_coeffs = self.spl_coeffs[0]
            pad_tmp = sp.interpolate_smooth(ref_coeffs, pad_v0, q)
            tmp = pad_tmp[pad_width:-pad_width, pad_width:-pad_width]
            self.ref_stack[q] = tmp[::scale, ::scale]

            pad_tmp = sp.interpolate_smooth(ref_coeffs, pad_v0, q, (1, 0))
            tmp = pad_tmp[pad_width:-pad_width, pad_width:-pad_width]
            self.ref_g_stack[q, :, :, 0] = tmp

            pad_tmp = sp.interpolate_smooth(ref_coeffs, pad_v0, q, (0, 1))
            tmp = pad_tmp[pad_width:-pad_width, pad_width:-pad_width]
            self.ref_g_stack[q, :, :, 1] = tmp

    def get_warp(self, p: int, q: int, w: np.ndarray, level=0):
        """
        Interpolate at the top scale.
        Parameters
        ----------
        p
        q: int
            assume q=0 is the top level here. Note that this is distinct to the
            scale or "level" because the scale is not always 0 in a ctf, but for
            q it is.
        w

        Returns
        -------
        """
        if p == 0:
            return self.ref_stack[q]

        w_height = w.shape[0]
        w_width = w.shape[1]

        scale = self.scale
        pad_width = self.pad_width

        if scale != 1:
            w_rescale = resample_2d(w, self.max_height, self.max_width)
        else:
            w_rescale = w
        v = np.zeros((self.max_height, self.max_height, 2))

        v[:, :, 0] = self.baselines[p, 0] * w_rescale
        v[:, :, 1] = self.baselines[p, 1] * w_rescale

        pad_v = np.pad(v, pad_width, mode='edge')[:, :, pad_width:-pad_width]
        c = self.spl_coeffs[p]

        if scale == 1:
            tmp = sp.interpolate_smooth(c, pad_v, q)
            i_pq = tmp[pad_width:-pad_width, pad_width:-pad_width]
        else:
            tmp = sp.interpolate_smooth(c, pad_v, q)
            tmp = tmp[pad_width:-pad_width, pad_width:-pad_width]
            i_pq = tmp[::scale, ::scale]

        if self.outside_image:
            f = np.empty((self.height, self.width, 2))
            f[:, :, 0] = w * self.baselines[p, 1]
            f[:, :, 1] = w * self.baselines[p, 0]
            i_pq = outside_image_option(i_pq, self.ref_stack[q], f)
        return i_pq

    def get_grad(self, p: int, q: int, w: np.ndarray, level=0):
        if p == 0:
            return self.ref_g_stack[q]

        scale = self.scale
        pad_width = self.pad_width
        w_height = w.shape[0]
        w_width = w.shape[1]

        if scale != 1:
            w_rescale = resample_2d(w, self.max_height, self.max_width)
        else:
            w_rescale = w
        v = np.zeros((self.max_height, self.max_height, 2))
        v[:, :, 0] = self.baselines[p, 0] * w_rescale
        v[:, :, 1] = self.baselines[p, 1] * w_rescale

        pad_v = np.pad(v, pad_width, mode='edge')[:, :, pad_width:-pad_width]
        c = self.spl_coeffs[p]

        if scale == 1:
            tmp = sp.interpolate_smooth(c, pad_v, q, (1, 0))
            Ipq_n = tmp[pad_width:-pad_width, pad_width:-pad_width]
            tmp = sp.interpolate_smooth(c, pad_v, q, (0, 1))
            Ipq_m = tmp[pad_width:-pad_width, pad_width:-pad_width]
        else:
            tmp = sp.interpolate_smooth(c, pad_v, q, (1, 0))
            g_res_y = tmp[pad_width:-pad_width, pad_width:-pad_width]
            tmp = sp.interpolate_smooth(c, pad_v, q, (0, 1))
            g_res_x = tmp[pad_width:-pad_width, pad_width:-pad_width]
            Ipq_n = g_res_y[::scale, ::scale]
            Ipq_m = g_res_x[::scale, ::scale]

        if self.outside_image:
            I_0q = self.ref_g_stack[q]
            I_0q_n = I_0q[:, :, 0]
            I_0q_m = I_0q[:, :, 1]
            f = np.empty((self.height, self.width, 2))
            f[:, :, 0] = w * self.baselines[p, 1]
            f[:, :, 1] = w * self.baselines[p, 0]
            Ipq_n = outside_image_option(Ipq_n, I_0q_n, f)
            Ipq_m = outside_image_option(Ipq_m, I_0q_m, f)

        nabla_Ipq = np.empty((self.height, self.width, 2))
        nabla_Ipq[:, :, 0] = Ipq_n * 2 ** q
        nabla_Ipq[:, :, 1] = Ipq_m * 2 ** q
        return nabla_Ipq


def hpf_gaussian(x: np.ndarray, q: int):
    """
    Basically uses a filter with taps defined by:
    h[n] = A * delta[n] - G_{sigma_q}[n]

    where,
    A = sum_{n} \ne {0} G_{sigma_q}[n]

    This is very similar to an unsharp mask.

    Parameters
    ----------
    x: np.ndarray
        Array to be filtered. Must be 2D.
    q: int
        The value which sets sigma_q

    Returns
    -------
    y: np.ndarray
        The result of convolution with the filter.
    """
    sigma = get_sigma(q)
    size = int(np.ceil(sigma * 4))  # 2.0/3.0 * 4
    ext = 2 * size + 1
    impulse = np.zeros((ext, ext))
    impulse[size, size] = 1
    kernel = -ndimage.gaussian_filter(impulse, sigma)
    kernel[size, size] = 0
    dc_gain = np.sum(kernel)
    kernel[size, size] = -dc_gain

    y = ndimage.convolve(x, kernel, mode='nearest')
    return y


def outside_image_option(w_img: np.ndarray, ref: np.ndarray, v: np.ndarray):
    """
    Replaces warping to areas outside the image with bits from the reference
    image
    Parameters
    ----------
    w_img
    ref
    v

    Returns
    -------

    """
    f = warp.flow_endpoints(v)
    height = w_img.shape[0]
    width = w_img.shape[1]
    f_x = f[:, :, 0]
    f_y = f[:, :, 1]
    out = np.where(0 <= f_x, w_img, ref)
    out = np.where(f_x < width, out, ref)
    out = np.where(0 <= f_y, out, ref)
    out = np.where(f_y < height, out, ref)
    return out


def resample_2d(arr, new_height, new_width, padtype='edge'):
    """
    Upscales w, using resample_poly

    Parameters
    ----------
    w
    new_height
    new_width
    padtype

    Returns
    -------

    """
    new_arr = signal.resample_poly(arr, new_height, arr.shape[0], axis=0,
                                   padtype='edge')
    new_arr = signal.resample_poly(new_arr, new_width, arr.shape[1], axis=1,
                                   padtype='edge')
    return new_arr


def radial_distortion(im: np.ndarray, K_1: float):
    """
    Applies radial distortion to a given image, assuming that the centre of the
    image is the distortion centre. We use the single term division model.

    Because cv.remap is output oriented, we calculate it based on where the
    inputs would be using this model.

    Assume that im is even in both dimensions.

    see: https://en.wikipedia.org/wiki/Distortion_(optics)#Software_correction

    Parameters
    ----------
    im: np.ndarray
        the image to distort
    K_1: float
        The distortion coefficient. We use a positive number for pincushion
        distortion and a negative number for barrel distortion.


    Returns
    -------
    np.ndarray
        the distorted image

    """
    import cv2 as cv

    height = im.shape[0]
    width = im.shape[1]

    K_1_val = K_1 / height ** 2

    y_c = height / 2 - 0.5
    x_c = width / 2 - 0.5

    y_d_temp = np.arange(0, height).astype(float)
    x_d_temp = np.arange(0, width).astype(float)

    x_d, y_d = np.meshgrid(x_d_temp, y_d_temp)

    r_sq = (x_d - x_c) ** 2 + (y_d - y_c) ** 2
    den = 1 + K_1_val * r_sq

    x_u = x_c + (x_d - x_c)/den
    y_u = y_c + (y_d - y_c)/den

    res = cv.remap(im, x_u.astype(np.float32), y_u.astype(np.float32),
                   cv.INTER_CUBIC)

    # plot.colorbar_img_plot(res)
    # plot.colorbar_img_plot(im)
    max_r_u_sq = (width - x_c)**2 + (height - y_c)**2
    num = 1 - np.sqrt(1 - 4 * K_1_val * max_r_u_sq)
    den = 2 * K_1_val * max_r_u_sq
    frac = num / den
    zoom_ratio = 1 / np.amin(frac)
    zoom_res = ndimage.zoom(res, zoom_ratio)
    z_height = zoom_res.shape[0]
    z_width = zoom_res.shape[1]
    # plot.colorbar_img_plot(zoom_res)
    #
    y_zc = int(np.round(z_height / 2.0))
    x_zc = int(np.round(z_width / 2.0))
    top = y_zc - int(height / 2)
    bot = y_zc + int(height / 2)
    left = int(x_zc - width / 2)
    right = int(x_zc + width / 2)

    out = zoom_res[top:bot, left:right]
    # plot.colorbar_img_plot(out)

    return out
