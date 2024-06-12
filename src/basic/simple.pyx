# Module simple.pyx
import math

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport ceil, floor, sqrt
from scipy import ndimage, signal

# DTYPE = np.float64
# ctypedef  np.float64_t DTYPE_t


derivative_kernel = np.array([0.5, 0., -0.5])

# sig = 2. / 3.
# size = int(ceil(sig * 4))  # 2.0/3.0 * 4
# x = np.zeros((2 * size + 1))
# x[size] = 1
# dog_kernel = ndimage.gaussian_filter1d(x, sig, order=1)
# dog_kernel = np.flip(dog_kernel)


def trunc_gaussian1d(
        np.ndarray[DTYPE_t, ndim=2] img,
        int axis,
        int direction,
        float sigma,
        str mode,
        cval=0.0
 ):
    """

    Parameters
    ----------
    img: np.ndarray
        image to be filtered
    axis: int
        0 is vertical, 1 is horizontal.
    direction: int
        0 is truncate so that the filter values are all left of the origin
        1 is the reverse.
    sigma
    mode
    cval

    Returns
    -------

    """
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    gaussian_kernel = ndimage.gaussian_filter1d(impulse, sigma)
    trunc_kernel = np.zeros_like(gaussian_kernel)
    if direction == 0:
        trunc_kernel[0:size + 1] = gaussian_kernel[0:size + 1]
    elif direction == 1:
        trunc_kernel[size:] = gaussian_kernel[size:]
    else:
        raise ValueError('Invalid Direction.')

    trunc_kernel /= np.sum(trunc_kernel)

    return ndimage.convolve1d(img, trunc_kernel, axis=axis, mode=mode,
                              cval=cval)


def trunc_gaussian2d(
        np.ndarray[DTYPE_t, ndim=2] img,
        int axis,
        int direction,
        float sigma,
        str mode,
        cval=0.0
):
    """

    Parameters
    ----------
    img: np.ndarray
        image to be filtered
    axis: int
        Axis to truncate 0 is vertical, 1 is horizontal.
    direction: int
        0 is truncate so that the filter values are all left of the origin
        1 is the reverse.
    sigma
    mode
    cval

    Returns
    -------

    """
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    gaussian_kernel = ndimage.gaussian_filter1d(impulse, sigma)
    trunc_kernel = np.zeros_like(impulse)
    if direction == 0:
        trunc_kernel[0:size + 1] = gaussian_kernel[0:size + 1]
    elif direction == 1:
        trunc_kernel[size:] = gaussian_kernel[size:]
    else:
        raise ValueError('Invalid Direction.')
    trunc_kernel /= np.sum(trunc_kernel)

    tmp = ndimage.convolve1d(img, trunc_kernel, axis=axis, mode=mode, cval=cval)
    other_axis = axis + 1 % 2
    # Axis = 0 becomes other_axis = 1, axis = 1 becomes other_axis = 0
    res = ndimage.convolve1d(img, trunc_kernel, axis=axis, mode=mode, cval=cval)
    return res


def dog_sq_filter1d(np.ndarray[DTYPE_t, ndim=2] img, int axis, float sigma,
                    str mode, cval=0.0):
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    dog_kernel = ndimage.gaussian_filter1d(impulse, sigma, order=1)
    dog_sq = dog_kernel ** 2
    res = filter1d(img, dog_sq, axis, mode, cval)
    return res


def gaussian_sq_filter2d(np.ndarray[DTYPE_t, ndim=2]img, float sigma, str mode,
                         cval=0.0):
    """
    Filters in both direction with a gaussian kernel squared.
    Parameters
    ----------
    img
    sigma
    mode
    cval

    Returns
    -------

    """
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    gaussian_kernel = ndimage.gaussian_filter1d(impulse, sigma)
    kernel = gaussian_kernel ** 2
    kernel /= np.sum(kernel)
    res = filter_sep2d(img, kernel, mode, cval)
    return res


def gaussian_filter2d(np.ndarray[DTYPE_t, ndim=2] img, float sigma, str mode,
                      cval=0.0):
    """
    Filters in both dimensions using a Gaussian kernel.
    Parameters
    ----------
    img
    sigma
    mode

    Returns
    -------

    """
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    gaussian_kernel = ndimage.gaussian_filter1d(impulse, sigma)
    return filter_sep2d(img, gaussian_kernel, mode, cval)


def derivative_of_gaussians1d(np.ndarray[DTYPE_t, ndim=2] img, int axis,
                              float sigma,
                              str mode, cval=0.0):
    """
    Performs a difference of gaussians along an axis.
    Parameters
    ----------
    img
    axis
    mode

    Returns
    -------

    """
    size = int(ceil(sigma * 4))  # 2.0/3.0 * 4
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    dog_kernel = ndimage.gaussian_filter1d(impulse, sigma, order=1)
    return filter1d(img, dog_kernel, axis, mode, cval)


def derivative1d(np.ndarray[DTYPE_t, ndim=2] img, int axis, str mode, cval=0.0):
    """

    Calculates the derivative of the img along one axis using derivative_kernel
    [0.5, 0., -0.5]

    Parameters
    ----------
    img
    axis
    mode

    Returns
    -------

    """
    return filter1d(img, derivative_kernel, axis, mode, cval)

def debug_filt_sep2d(img, kernel, mode, cval=0.0):
    return filter_sep2d(img, kernel, mode, cval)


cdef filter_sep2d(np.ndarray[DTYPE_t, ndim=2] img,
                  np.ndarray[DTYPE_t, ndim=1] kernel,
                  str mode, float cval):
    """
    Filters in both directions using a separable kernel.

    Parameters
    ----------
    cval
    img
    kernel
    axis
    mode
    """
    cdef int len_kernel = kernel.shape[0]
    if len_kernel % 2 == 0:
        raise ValueError('Length of Kernel should be odd')
    #
    # cdef int border = (len_kernel - 1) // 2

    if mode == 'edge':
        tmp = ndimage.convolve1d(img, kernel, axis=1, mode='nearest', cval=cval)
        out = ndimage.convolve1d(tmp, kernel, axis=0, mode='nearest', cval=cval)
    else:
        tmp = ndimage.convolve1d(img, kernel, axis=1, mode=mode, cval=cval)
        out = ndimage.convolve1d(tmp, kernel, axis=0, mode=mode, cval=cval)

    # tmp = signal.upfirdn(kernel, img, axis=0, mode=mode, cval=cval)
    # big_res = signal.upfirdn(kernel, tmp, axis=1, mode=mode, cval=cval)
    # res = big_res[border:-border, border:-border]
    return out


cdef filter1d(np.ndarray[DTYPE_t, ndim=2] img,
              np.ndarray[DTYPE_t, ndim=1] kernel,
              int axis, str mode, float cval):
    """
    Do Gaussian filtering on the
    Parameters
    ----------

    img
    kernel:
        should be of odd length
    axis
    mode
    cval

    Returns
    -------

    """

    cdef int len_kernel = kernel.shape[0]
    if len_kernel % 2 == 0:
        raise ValueError('Length of Kernel should be odd')

    if axis > 1 or axis < 0:
        raise ValueError('Axis should be 0 (x) or 1 (y)')

    cdef int border = (len_kernel - 1) // 2

    res_border = signal.upfirdn(kernel, img, axis=axis, mode=mode, cval=cval)
    if axis == 0:
        filtered_res = res_border[border:-border, :]
    else:
        filtered_res = res_border[:, border:-border]

    return filtered_res


def debug_filter2d(img, kernel, mode, cval=0.0):
    return filter2d(img, kernel, mode, cval)


cdef np.ndarray[DTYPE_t, ndim=2] filter2d(np.ndarray[DTYPE_t, ndim=2] img,
                                          np.ndarray[DTYPE_t, ndim=2] kernel,
                                          str mode, float cval):
    cdef int h_kernel = kernel.shape[0]
    if h_kernel % 2 == 0:
        raise ValueError('height of Kernel should be odd')

    cdef int w_kernel = kernel.shape[1]
    if w_kernel % 2 == 0:
        raise ValueError('width of Kernel should be odd')

    cdef int pad_width = int(max((h_kernel - 1) / 2, (w_kernel - 1) / 2))

    if mode == 'antireflect':
        pad_img = np.pad(img, pad_width, mode='reflect', reflect_type='odd')
    elif mode =='constant':
        pad_img = np.pad(img, pad_width, mode=mode, constant_values=cval)
    else:
        pad_img = np.pad(img, pad_width, mode=mode)

    res = signal.convolve2d(pad_img, kernel, mode='valid')
    return res


@cython.cdivision(True)  # Use C-division where we don't check divide by zero error
def resample_img(np.ndarray[DTYPE_t, ndim=2] img, int up, int down,
                 window=('kaiser', 5.0), mode='antireflect'):
    """

    Parameters
    ----------
    img
    up
    down
    window

    Returns
    -------

    """
    if up == 0 or down == 0:
        raise ValueError('Infinite or zero resampling does not make sense')

    cdef int height
    cdef int width
    height = img.shape[0]
    width = img.shape[1]

    cdef int max_rate = max(up, down)
    cdef double f_c = 1. / max_rate
    cdef int half_len = 10 * max_rate
    h = np.array(signal.firwin(2 * half_len + 1, f_c, window=window)) * up

    output = signal.upfirdn(h, img, up, down, axis=0, mode=mode)
    output = signal.upfirdn(h, output, up, down, axis=1, mode=mode)

    cdef int new_height = int(height * up / down)
    cdef int new_width = int(width * up / down)
    cdef int n_pre_pad = (down - half_len % down)
    cdef int pre_remove = (half_len + n_pre_pad) // down

    output = output[pre_remove - 1:new_height + pre_remove - 1,
             pre_remove - 1:new_width + pre_remove - 1]
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef weighted_sad(
    np.ndarray[DTYPE_t, ndim=2] x: np.ndarray,
    np.ndarray[DTYPE_t, ndim=2] weights: np.ndarray
):
    """
    Weighted sum of absolute differences between x[n] and its neighbours within
    a neighbourhood of weights. 

    At the edges, we will do zero-order hold boundary extension.

    Parameters
    ----------
    x: np.ndarray
        must be 2D.
    weights: np.ndarray
        must be 2D and each dimension must be odd in length. We only look at the
        non-centre values. Essentially these are the weights of the differences.

    Returns
    -------
    y: np.ndarray
        The calculated sum of absolute differences.
    """
    cdef int w_height = weights.shape[0]
    cdef int w_width = weights.shape[1]
    cdef int half_h = int((w_height - 1) / 2)
    cdef int half_w = int((w_width - 1) / 2)
    cdef int pad_size = int((np.maximum(half_w, half_h)))
    abs_weights = np.abs(weights)

    pad_x = np.pad(x, pad_size, mode='edge')
    cdef int height = x.shape[0]
    cdef int width = x.shape[1]
    y = np.empty_like(x)

    cdef int r, c, r_lim, c_lim, w_r, w_c
    cdef double centre

    for r in range(height):
        for c in range(width):
            r_lim = r + w_height
            c_lim = c + w_width
            centre = x[r, c]
            diff = np.abs(pad_x[r:r_lim, c:c_lim] - centre)
            y[r, c] = np.sum(diff * weights)

    return y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef weighted_ssd(
    np.ndarray[DTYPE_t, ndim=2] x: np.ndarray,
    np.ndarray[DTYPE_t, ndim=2] weights: np.ndarray
):
    """
    Weighted sum of absolute differences between x[n] and its neighbours within
    a neighbourhood of weights. 

    At the edges, we will do zero-order hold boundary extension.

    Parameters
    ----------
    x: np.ndarray
        must be 2D.
    weights: np.ndarray
        must be 2D and each dimension must be odd in length. We only look at the
        non-centre values. Essentially these are the weights of the differences.

    Returns
    -------
    y: np.ndarray
        The calculated sum of absolute differences.
    """
    cdef int w_height = weights.shape[0]
    cdef int w_width = weights.shape[1]
    cdef int half_h = int((w_height - 1) / 2)
    cdef int half_w = int((w_width - 1) / 2)
    cdef int pad_size = int((np.maximum(half_w, half_h)))
    abs_weights = np.abs(weights)

    pad_x = np.pad(x, pad_size, mode='edge')
    cdef int height = x.shape[0]
    cdef int width = x.shape[1]
    y = np.empty_like(x)

    cdef int r, c, r_lim, c_lim, w_r, w_c
    cdef double centre

    for r in range(height):
        for c in range(width):
            r_lim = r + w_height
            c_lim = c + w_width
            centre = x[r, c]
            diff = (pad_x[r:r_lim, c:c_lim] - centre) ** 2
            y[r, c] = np.sum(diff * weights)

    return y


# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef local_max_diff(
    np.ndarray[DTYPE_t, ndim=2] a: np.ndarray,
    double radius: float
):
    """
    Maximum of the absolute differences between x[n] and its neighbours within
    radius

    At the edges, we will do zero-order hold boundary extension.

    Parameters
    ----------
    a: np.ndarray
        must be 2D.
    radius: float 
        size of a circular disk around x[n]

    Returns
    -------
    y: np.ndarray
        The calculated sum of absolute differences.
    """

    cdef int pad_size = int(math.ceil(radius))
    ext = 2 * pad_size + 1
    x, y = np.indices((ext, ext))
    weights = np.where(np.hypot(radius - x, radius - y) < radius, 1.0, 0.0)

    pad_a = np.pad(a, pad_size, mode='edge')
    cdef int height = a.shape[0]
    cdef int width = a.shape[1]
    y = np.empty_like(a)

    cdef int r, c, r_lim, c_lim, w_r, w_c
    cdef double centre

    for r in range(height):
        for c in range(width):
            r_lim = r + ext
            c_lim = c + ext
            centre = a[r, c]
            diff = np.abs(pad_a[r:r_lim, c:c_lim] - centre)
            y[r, c] = np.amax(diff * weights)

    return y

def debug_filter1d(img, kernel, axis, mode, cval=0.0):
    """
    Just for testing
    Parameters
    ----------
    img
    kernel
    axis
    mode

    Returns
    -------

    """
    return filter1d(img, kernel, axis, mode, cval)


cpdef np.ndarray[DTYPE_t, ndim=2] gaussian_win_var(np.ndarray[DTYPE_t, ndim=2] x,
                                                   float sigma, str mode):
    """
    Calculates the variance with a gaussian window. Uses equation
    Var[X] = E[X^2] -  E[X]^2

    Parameters
    ----------
    x: np.ndarray
        Array to calculate the variance over
    sigma: float
        Sigma value for the gaussian
    mode: str, optional
        The mode parameter determines how the input array is extended when the
        filter overlaps a border. By passing a sequence of modes with length
        equal to the number of dimensions of the input array, different modes
        can be specified along each axis. Default value is ‘reflect’. The valid
        values and their behavior is as follows:

        ‘reflect’ (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last
            pixel. This mode is also sometimes referred to as half-sample
            symmetric.

        ‘constant’ (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the
            same constant value, defined by the cval parameter.
        ‘nearest’ (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.
        ‘mirror’ (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last
            pixel.  This mode is also sometimes referred to as whole-sample
            symmetric.
        ‘wrap’ (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.

        For consistency with the interpolation functions, the following mode
        names can also be used:

        ‘grid-constant’
            This is a synonym for ‘constant’.
        ‘grid-mirror’
            This is a synonym for ‘reflect’.
        ‘grid-wrap’
            This is a synonym for ‘wrap’.



    Returns
    -------
    np.ndarray
        Same shape as x.

    """
    e_x = ndimage.gaussian_filter(x, sigma, mode=mode)
    e_x2 = ndimage.gaussian_filter(x ** 2, sigma, mode=mode)

    var = e_x2 - e_x ** 2

    return var


cpdef np.ndarray[DTYPE_t, ndim=2] gauss_win_xvar(np.ndarray[DTYPE_t, ndim=2] x,
                                                 np.ndarray[DTYPE_t, ndim=2] y,
                                                 float sigma, str mode):
    """
    Var[X] = E[XY] -  E[X]E[Y]
    
    Parameters
    ----------
    x: np.ndarray
        Array to calculate the variance over
    y: np.ndarray
        Array to calculate the variance over
    sigma: float
        Sigma value for the gaussian
    mode: str, optional
        The mode parameter determines how the input array is extended when the
        filter overlaps a border. By passing a sequence of modes with length
        equal to the number of dimensions of the input array, different modes
        can be specified along each axis. Default value is ‘reflect’. The valid
        values and their behavior is as follows:

        ‘reflect’ (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last
            pixel. This mode is also sometimes referred to as half-sample
            symmetric.

        ‘constant’ (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the
            same constant value, defined by the cval parameter.
        ‘nearest’ (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.
        ‘mirror’ (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last
            pixel.  This mode is also sometimes referred to as whole-sample
            symmetric.
        ‘wrap’ (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.

        For consistency with the interpolation functions, the following mode
        names can also be used:

        ‘grid-constant’
            This is a synonym for ‘constant’.
        ‘grid-mirror’
            This is a synonym for ‘reflect’.
        ‘grid-wrap’
            This is a synonym for ‘wrap’.


    Returns
    -------

    """
    if y.shape[0] != x.shape[0]:
        raise ValueError("X and Y are different sizes")
    if y.shape[1] != x.shape[1]:
        raise ValueError("X and Y are different sizes")

    e_x = ndimage.gaussian_filter(x, sigma, mode=mode)
    e_y = ndimage.gaussian_filter(y, sigma, mode=mode)

    e_xy = ndimage.gaussian_filter(x * y, sigma, mode=mode)
    return e_xy - e_x * e_y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] magnitude(double[:, :, :] x):
    """
    Speeds up the operation 
    np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
    
    Parameters
    ----------
    x

    Returns
    -------

    """
    cdef int height = x.shape[0]
    cdef int width = x.shape[1]
    cdef int row, col

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((height, width))
    cdef double[:, :] y = out

    for row in range(0, height):
        for col in range(0, width):
            y[row, col] = sqrt(x[row, col, 0] ** 2 + x[row, col, 1] ** 2)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] mag_sq(double[:, :, :] x):
    """
    Hopefully speeds up the operation 
    x[:, :, 0] ** 2 + x[:, :, 1] ** 2

    Parameters
    ----------
    x

    Returns
    -------

    """
    cdef int height = x.shape[0]
    cdef int width = x.shape[1]
    cdef int row, col

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((height, width))
    cdef double[:, :] y = out

    for row in range(0, height):
        for col in range(0, width):
            y[row, col] = x[row, col, 0] ** 2 + x[row, col, 1] ** 2

    return out



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] sum2d(
    double[:, ::1] x,
    double[:, ::1] y
):
    """
    Does x + y. Both must be the same size
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    cdef int height = x.shape[0]
    cdef int width = x.shape[1]

    if y.shape[0] != height:
        raise ValueError("X and Y are different sizes")
    if y.shape[1] != width:
        raise ValueError("X and Y are different sizes")

    cdef int row, col
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.empty((height, width))
    cdef double[:, :] z = output


    for row in range(0, height):
        for col in range(0, width):
            z[row, col] = x[row, col] + y[row, col]

    return output


cpdef np.ndarray[DTYPE_t, ndim=3] weight_gradient(np.ndarray[DTYPE_t, ndim=3] grad,
                                                  np.ndarray[DTYPE_t, ndim=2] weights):
    """

    Parameters
    ----------
    grad: np.ndarray
        [height, width, 2]
    weights:
        [height, width]

    Returns
    -------

    """
    cdef np.ndarray[DTYPE_t, ndim=3] weighted_gradient = np.empty_like(grad)
    weighted_gradient[:, :, 0] = grad[:, :, 0] * weights
    weighted_gradient[:, :, 1] = grad[:, :, 1] * weights

    return weighted_gradient


cpdef baseline_mag_sq(baseline):
    return baseline[0] ** 2 + baseline[1] ** 2


cpdef rmse(gt_disparity: np.ndarray, est_disparity: np.ndarray):
    """
    Root-mean-square of the difference between the disparity and the ground
    truth
    Returns
    -------

    """

    diff = gt_disparity - est_disparity
    return np.sqrt(np.mean(diff ** 2))


cpdef pred_err(disparity: np.ndarray, baseline: np.ndarray | tuple,
               ref: np.ndarray, tgt: np.ndarray, dir='fwd', mode='rms'):
    """
    
    Parameters
    ----------
    disparity
    baseline
    ref
    tgt
    dir
    mode

    Returns
    -------

    """

    cdef int height = disparity.shape[0]
    cdef int width = disparity.shape[1]

    if np.allclose(disparity, 0) or baseline == (0, 0):
        err = ref - tgt
        if mode == 'rms':
            return np.sqrt(np.nanmean(err ** 2))
        elif mode == 'abs':
            return np.mean(np.abs(err))
        else:
            raise ValueError('Invalid mode supplied.')
    else:
        d_flows = np.empty((height, width, 2))
        d_flows[:, :, 0] = disparity * baseline[0]
        d_flows[:, :, 1] = disparity * baseline[1]
        err = flow_pred_err(d_flows, ref, tgt, dir, mode)
    return err


cpdef flow_pred_err(flows: np.ndarray, ref: np.ndarray, tgt: np.ndarray,
                    dir='fwd', mode='rms'):
    """

    Warps target to look like the reference view. Then we evaluate how accurate
    that is.

    Parameters
    ----------
    flows
    ref
    tgt
    dir
    mode

    Returns
    -------

    """
    from lowlevel import warp

    if dir == 'fwd':
        w_tgt = warp.fast_warp_upscale(tgt, flows, 'edge')
        err = ref - w_tgt
    elif dir == 'bwd':
        w_ref = warp.fast_warp_upscale(ref, -flows, 'edge')
        err = w_ref - tgt
    else:
        raise ValueError('Invalid direction supplied')

    if mode == 'rms':
        score = np.sqrt(np.nanmean(err ** 2))
    elif mode == 'abs':
        score = np.nanmean(np.abs(err))
    else:
        raise ValueError('Invalid mode supplied.')

    return score


cpdef int tuple_search(arr: np.ndarray, t: tuple | np.ndarray):
    """
    Searches arr for t. arr must be a 2D array with dimensions:
    [n, len(t)]

    Parameters
    ----------
    arr
    t

    Returns
    -------
    int
        the index where t is. if it's -1, we haven't found it
        
    Notes
    -----
    
    Because we probably be using this to find the reference view on sorted 
    arrays, this is probably a fairly optimal solution. The first view would
    likely be the reference view.

    """
    cdef int t_len = len(t)
    cdef int a_len = arr.shape[0]
    cdef int stride = arr.shape[1]

    if stride != t_len:
        raise ValueError('arr and t have a dimension mismatch.')

    cdef int n
    for n in range(a_len):
        val = arr[n]
        if np.all(val == t):
            return n

    return -1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_t, ndim=2] valid_data(double[:, :, :] flows):
    """
    Determines whether invalid data from outside the image boundaries has been
    warped into the image in at each point in the image.

    This sort of is like reversing the flow field and then determining which 
    points in the target image would be from outside the image accordingly.
    Remember, the flow field is from reference to target. 

    Valid data is a 1, invalid data is a 0.

    Parameters
    ----------
    flows

    Returns
    -------

    """
    cdef int height = flows.shape[0]
    cdef int width = flows.shape[1]

    cdef double max_flow = max_magnitude3d(flows)

    cdef int b_size = int(round(max_flow))

    if b_size < 1:
        return np.ones((height, width))

    cdef int bottom_border = height - b_size
    cdef int right_border = width - b_size

    valid_arr = np.empty((height, width))
    cdef double[:, :] valid = valid_arr
    cdef int r, c
    cdef double flow_x, flow_y

    for r in range(0, height):
        for c in range(0, width):
            if r < b_size or r >= bottom_border or \
                    c < b_size or c >= right_border:
                flow_x = round(flows[r, c, 0]) + c
                flow_y = round(flows[r, c, 1]) + r
                if 0 <= flow_x < width and 0 <= flow_y < height :
                    valid[r, c] = 1
                else:
                    valid[r, c] = 0
            else:
                valid[r, c] = 1

    return valid_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cdef double max_magnitude3d(double[:, :, :] flows):
    cdef int height = flows.shape[0]
    cdef int width = flows.shape[1]

    cdef double max_flow = -1

    cdef int row, col
    cdef double flow_magnitude, flow_x, flow_y
    for row in range(height):
        for col in range(width):
            flow_x = flows[row, col, 0]
            flow_y = flows[row, col, 1]
            flow_magnitude = sqrt(flow_x ** 2 + flow_y ** 2)
            if max_flow < flow_magnitude:
                max_flow = flow_magnitude

    return max_flow


cpdef double get_sigma(int q):
    """
    sigma = sqrt(2) * 2 ** q / 2
    
    Parameters
    ----------
    q: int

    Returns
    -------
    sigma
    """
    cdef double sigma = sqrt(2) * 2. ** float(q) / 2.
    # cdef double sigma = 2. * 2 ** q / 3.
    return sigma
