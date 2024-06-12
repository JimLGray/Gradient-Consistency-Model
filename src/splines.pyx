# Module splines.py
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fmax, fabs, floor, ceil, round
from scipy import ndimage
from simple import get_sigma

DTYPE = np.float64
ctypedef  np.float64_t DTYPE_t

"""
The spline we are using is defined as
$$
\beta ^{3}(x) = \left\{ \begin{array}{ll}
\frac{2}{3} - |x|^2 + \frac{|x|^3}{2},  & 0 \leq |x| < 1 \\
\frac{(2 - |x|)^3}{6}, & 1 \leq |x| < 2 \\
0, & 2 \leq |x|
\end{array}\right.
$$

Note any interactions with these splines need padding by 1 for proper 
invertability

"""

z_1 = -2 + np.sqrt(3)
const = (z_1 / (1 - z_1 ** 2))

# Pre-Flipping the axis.
x_arr = np.arange(2., -3., -1.)
# length 5
filt_n = np.empty(5, dtype=DTYPE)
filt_m = np.empty(5, dtype=DTYPE)
x_n_arr = np.empty(5, dtype=DTYPE)
x_m_arr = np.empty(5, dtype=DTYPE)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cpdef interpolate(c: np.ndarray, v: np.ndarray, order=(0, 0),
                boundary_cond='edge', n_dims=2):
    """
    Performs cubic B-spline interpolation using a series of spline coefficients.
    The shifts (v) are referenced such that they are a series of vectors that
    when added to their location point to the location to interpolate.
    Expressing this mathematically
    y[n] = x[n + v[n]]

    For example shift[1, 2] has value [3, 4], this means that the location
    that is interpolated is [3, 6].

    We do integer shifting first and then use interpolation to perform the
    fractional shifts.

    Parameters
    ----------
    c: np.ndarray
        The spline coefficients used to do the interpolation
    v: np.ndarray
        So this has to be one dimension higher than s, I think.
        So if c is [height, width]. v has to be [height, width, 2]
        where the last two dimensions directly point out the area to be
        interpolated. These are vectors that are like flow fields essentially.
        The last dimension also has to have a size equal to the number of
        dimensions of c
    order: tuple
        0th order refers to 0th derivative in that direction.
        1st order refers to 1st derivative in that direction.
        We use (y, x) notation. If c is 1d, we use the first item in order only.
        This is the same notation as the default numpy array.
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.
    ndims: int
        number of dimensions.

    Returns
    -------
    y: np.ndarray
        The interpolated result. Same dimensions as c
    """
    y = np.empty_like(c)
    v_int = np.round(v).astype(int)
    frac_v = v - v_int
    cdef int pad_width = int(np.amax(np.ceil(np.abs(v))) + 2)
    pad_c = np.pad(c, pad_width, mode=boundary_cond)
    # Need to think about this.
    cdef int n, m, N, M, v_int_n, s_idx
    cdef double v_n

    if n_dims == 1:
        N = c.shape[0]
        for n in range(N):
            x = np.arange(-2., 3.)
            # The idea is that first we find the location in the array with the
            # closest integer value. Then we find the remaining fractional bit.
            # After that we do a weighted sum of the c values in a local area
            # the weights are the shifted kernels. The kernels need to be
            # flipped to do the convolution properly.
            v_int_n = v_int[n, 0]
            v_n = frac_v[n, 0]
            s_idx = n + v_int_n + pad_width
            local_c = pad_c[s_idx-2:s_idx+3]
            if order[0] == 0:
                filt_arr = np.flip(cubic_b_spline_1d(x + v_n))
            elif order[0] == 1:
                filt_arr = np.flip(derivative_cubic_spline1d(x + v_n))
            else:
                raise ValueError('Unsupported order.')
            y[n] = np.sum(local_c * filt_arr)

    elif n_dims == 2:
        N = c.shape[0]
        M = c.shape[1]
        interpolate2d(pad_c, y, v_int, frac_v, order[0], order[1], N, M,
                      pad_width)

    return y

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# @cython.cdivision(True) # Use C-division where we don't check divide by zero error
cpdef np.ndarray[DTYPE_t, ndim=1] upscale_interpolate1d(
        np.ndarray[DTYPE_t, ndim=1] c,
        np.ndarray[DTYPE_t, ndim=2] v,
        int scale,
        order=0,
        boundary_cond='edge'
):
    """
    Upscales and interpolates using spline coefficients in 1D. The amount of
    interpolation is determined by the size of y, it is assumed to be roughly
    an integer multiple of c in size.

    Parameters
    ----------
    c
    v
    scale
    order
    boundary_cond

    Returns
    -------
    y: np.ndarray
        upscaled output,
    """
    cdef int c_len = c.shape[0]
    cdef int y_len = c_len * scale
    cdef int v_len = v.shape[0]

    if v_len != y_len:
        raise ValueError('Mismatch in Dimensions')
    y_arr = np.empty(y_len)
    cdef double[:] y = y_arr
    cdef int pad_width = int(np.amax(np.ceil(np.abs(v/scale) + 0.5)) + 2)
    pad_c_arr = np.pad(c, pad_width, mode=boundary_cond)
    cdef double[:] pad_c = pad_c_arr

    cdef double[:] f_n = filt_n
    cdef int length_x = 5
    cdef double[:] x_n = x_n_arr
    cdef double[:] x = x_arr
    cdef double v_n, total, upscale_shift, vfrac_n
    cdef double n_shift, f_n_val

    cdef int order_0 = order

    cdef int i, n, v_int_n, s_n, n_idx, s_n_idx

    for n in range(y_len):
        n_idx = int(n / scale)
        n_shift = float(n) / float(scale) - n_idx
        v_n = v[n, 0] / float(scale) + n_shift
        v_int_n = int(v_n)
        vfrac_n = v_n - v_int_n
        s_n = int(n_idx + v_int_n + pad_width)

        for i in range(length_x):
            x_n[i] = x[i] + vfrac_n

        if order_0 == 0:
            _cubic_b_spline_1d(x_n, f_n, length_x)
        else:
            _derivative_cubic_spline1d(x_n, f_n, length_x)

            total = 0.
            for i in range(-2, 3):
                f_n_val = f_n[i + 2]
                s_n_idx = s_n + i
                total += f_n_val * pad_c[s_n_idx]

            y[n] = total

    return y_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cpdef np.ndarray[DTYPE_t, ndim=2] upscale_interpolate2d(
        np.ndarray[DTYPE_t, ndim=2] c,
        np.ndarray[DTYPE_t, ndim=3] v,
        int scale,
        order=(0, 0),
        boundary_cond='edge'
):
    """
    Upscales and interpolates using spline coefficients. The amount of
    interpolation is determined by the size of y, it is assumed to be roughly
    an integer multiple of c in size.

    Parameters
    ----------
    c: np.ndarray
        spline coefficients 2d array
    v: np.ndarray
        flow vector field, must be equal in size to 
        [c_height * scale, c_width * scale, 2]
    scale: int
        scaling factor
    order: tuple
        0th order refers to 0th derivative in that direction.
        1st order refers to 1st derivative in that direction.
        We use (y, x) notation. If c is 1d, we use the first item in order only.
        This is the same notation as the default numpy array.
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.

    Returns
    -------
    y: np.ndarray
        upscaled output,

    """
    cdef int c_height = c.shape[0]
    cdef int c_width = c.shape[1]

    # cdef int y_height = c_height * scale
    # cdef int y_width = c_width * scale

    cdef int v_height = v.shape[0]
    cdef int v_width = v.shape[1]

    # if y_width != v_width or y_height != v_height:
    if int(np.ceil(float(v_height) / float(scale))) != c_height:
        # print(v_height / scale)
        raise ValueError('Mismatch in Height')
    if int(np.ceil(float(v_width) / float(scale))) != c_width:
        raise ValueError('Mismatch in Width')

    y_arr = np.empty((v_height, v_width))
    cdef double[:, :] y = y_arr

    # need to scale v appropriately here...
    cdef int pad_width = int(np.amax(np.ceil(np.abs(v/scale) + 0.5)) + 2)
    pad_c_arr = np.pad(c, pad_width, mode=boundary_cond)
    cdef double[:, :] pad_c = pad_c_arr

    cdef double[:] f_n = filt_n
    cdef double[:] f_m = filt_m
    cdef int length_x = 5

    cdef double[:] x_n = x_n_arr
    cdef double[:] x_m = x_m_arr
    cdef double[:] x = x_arr

    cdef double v_n, v_m, total, upscale_shift, mul, vfrac_n, vfrac_m
    cdef double m_shift, n_shift, f_n_val

    cdef int order_0 = order[0]
    cdef int order_1 = order[1]

    cdef int i, j, n, m, v_int_n, v_int_m, s_n, s_m, n_idx, m_idx, s_n_idx

    for n in range(v_height):
        for m in range(v_width):
            # Calculate scale based shift...
            n_idx = int(n / scale)
            m_idx = int(m / scale)
            n_shift = float(n) / float(scale) - n_idx
            m_shift = float(m) / float(scale) - m_idx
            v_n = v[n, m, 0] / float(scale) + n_shift
            v_m = v[n, m, 1] / float(scale) + m_shift
            v_int_n = int(v_n)
            v_int_m = int(v_m)
            vfrac_n = v_n - v_int_n
            vfrac_m = v_m - v_int_m

            s_n = int(n_idx + v_int_n + pad_width)
            s_m = int(m_idx + v_int_m + pad_width)

            for i in range(length_x):
                x_n[i] = x[i] + vfrac_n
                x_m[i] = x[i] + vfrac_m

            if order_0 == 0:
                _cubic_b_spline_1d(x_n, f_n, length_x)
            else:
                _derivative_cubic_spline1d(x_n, f_n, length_x)
            if order_1 == 0:
                _cubic_b_spline_1d(x_m, f_m, length_x)
            else:
                _derivative_cubic_spline1d(x_m, f_m, length_x)

            total = 0.
            for i in range(-2, 3):
                f_n_val = f_n[i + 2]
                s_n_idx = s_n + i
                for j in range(-2, 3):
                    mul = f_n_val * f_m[j + 2]
                    total += mul * pad_c[s_n_idx, s_m + j]

            y[n, m] = total

    return y_arr

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cpdef np.ndarray[DTYPE_t, ndim=2] interpolate_smooth(
        np.ndarray[DTYPE_t, ndim=2] c,
        np.ndarray[DTYPE_t, ndim=3] v,
        int q,
        order=(0, 0),
        boundary_cond='edge'
):
    """
    We perform the interpolation and smoothing at the same stage. We take the 
    interpolation kernel and convolve it with the appropriate gaussian filter
    for that scale sigma = sqrt(2) * 2 ** q / 2

    Parameters
    ----------
    c: np.ndarray
        spline coefficients 2d array
    v: np.ndarray
        flow vector field, must be equal in size to 
        [c_height * scale, c_width * scale, 2]
    q: int
        the scale we are operating at. q = 0 is the finest scale. 
    order: tuple
        0th order refers to 0th derivative in that direction.
        1st order refers to 1st derivative in that direction.
        We use (y, x) notation. If c is 1d, we use the first item in order only.
        This is the same notation as the default numpy array.
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.

    Returns
    -------
    y: np.ndarray
        upscaled output,

    """
    cdef int c_height = c.shape[0]
    cdef int c_width = c.shape[1]

    cdef int v_height = v.shape[0]
    cdef int v_width = v.shape[1]

    # if y_width != v_width or y_height != v_height:
    if v_height != c_height:
        # print(v_height / scale)
        raise ValueError('Mismatch in Height')
    if v_width != c_width:
        raise ValueError('Mismatch in Width')

    y_arr = np.empty((v_height, v_width))
    cdef double[:, :] y = y_arr

    # need to scale v appropriately here...
    cdef int max_v = int(np.amax(np.ceil(np.abs(v))))
    cdef float sigma = get_sigma(q)
    # cdef int gauss_ext = int(np.ceil(sigma * 4))
    cdef int filt_ext = 2

    cdef int pad_width = max_v + filt_ext
    pad_c_arr = np.pad(c, pad_width, mode=boundary_cond)
    cdef double[:, :] pad_c = pad_c_arr

    cdef int length_x = 2 * filt_ext + 1

    filtbank_n = np.empty(length_x)
    filtbank_m = np.empty(length_x)
    cdef double[:] f_n = filtbank_n
    cdef double[:] f_m = filtbank_m

    xn_arr = np.empty(length_x)
    xm_arr = np.empty(length_x)
    # pre-flip x_axis

    cdef double[:] x_n = xn_arr
    cdef double[:] x_m = xm_arr
    cdef double[:] x = x_arr

    cdef double v_n, v_m, total, mul, vfrac_n, vfrac_m
    cdef double f_n_val, total_n, total_m

    cdef int order_0 = order[0]
    cdef int order_1 = order[1]

    cdef int i, j, n, m, v_int_n, v_int_m, s_n, s_m, s_n_idx
    impulse = np.zeros(length_x)
    impulse[filt_ext] = 1.0

    for n in range(v_height):
        for m in range(v_width):
            # Calculate scale based shift...
            v_n = v[n, m, 0]
            v_m = v[n, m, 1]
            v_int_n = int(v_n)
            v_int_m = int(v_m)
            vfrac_n = v_n - v_int_n
            vfrac_m = v_m - v_int_m

            s_n = int(n + v_int_n + pad_width)
            s_m = int(m + v_int_m + pad_width)

            for i in range(5):
                x_n[i] = x[i] + vfrac_n
                x_m[i] = x[i] + vfrac_m

            _cubic_b_spline_1d(x_n, f_n, length_x)
            _cubic_b_spline_1d(x_m, f_m, length_x)

            # if order_0 == 0:
            #     _cubic_b_spline_1d(x_n, f_n, length_x)
            # else:
            #     _derivative_cubic_spline1d(x_n, f_n, length_x)
            # if order_1 == 0:
            #     _cubic_b_spline_1d(x_m, f_m, length_x)
            # else:
            #     _derivative_cubic_spline1d(x_m, f_m, length_x)

            total = 0.
            for i in range(-filt_ext, filt_ext + 1):
                f_n_val = f_n[i + filt_ext]
                s_n_idx = s_n + i
                for j in range(-filt_ext, filt_ext + 1):
                    mul = f_n_val * f_m[j + filt_ext]
                    total += mul * pad_c[s_n_idx, s_m + j]

            y[n, m] = total

    y_arr = ndimage.gaussian_filter(y_arr, sigma, order, mode='nearest')

    return y_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void interpolate2d(
        double[:, ::1] c,
        double[:, ::1] y,
        long[:, :, ::1] v_int,
        double[:, :, ::1] v_frac,
        int order_0,
        int order_1,
        int N,
        int M,
        int pad_width
):
    """
    
    Parameters
    ----------
    c: double[:, :]
        the spline coefficients to interpolate. Boundary must be extended 
        properly as per interpolate 
    y: double[:, :]
        the interpolation results
    v_int: double[:, :, :]
        the integer shifts
    v_frac: double[:, :, :]
        the integer shifts
    order_0: int
        0 is no derivative, 1 is derivative
    order_1: int
        0 is no derivative, 1 is derivative
    N: int 
        Length in the n direction (generally height)
    M: int
        Length in the m direction (generally width)

    Returns
    -------
    y: double[:, :]
        The interpolated data.
    """
    cdef int n, m, s_n, s_m, i, j
    cdef double v_int_n, v_int_m
    cdef double v_n, v_m
    cdef double mul, total

    cdef int length_x = 5
    cdef double[:] x = x_arr
    cdef double[:] x_n = x_n_arr
    cdef double[:] x_m = x_m_arr

    cdef double[:] f_n_view = filt_n
    cdef double[:] f_m_view = filt_m

    for n in range(N):
        for m in range(M):
            v_int_n = v_int[n, m, 0]
            v_int_m = v_int[n, m, 1]
            v_n = v_frac[n, m, 0]
            v_m = v_frac[n, m, 1]
            s_n = int(n + v_int_n + pad_width)
            s_m = int(m + v_int_m + pad_width)
            for i in range(length_x):
                x_n[i] = x[i] + v_n
                x_m[i] = x[i] + v_m

            if order_0 == 0:
                _cubic_b_spline_1d(x_n, f_n_view, length_x)
            elif order_0 == 1:
                _derivative_cubic_spline1d(x_n, f_n_view, length_x)
            else:
                raise ValueError('Unsupported order.')
            if order_1 == 0:
                _cubic_b_spline_1d(x_m, f_m_view, length_x)
            elif order_1 == 1:
                _derivative_cubic_spline1d(x_m, f_m_view, length_x)
            else:
                raise ValueError('Unsupported order.')
            total = 0.

            for i in range(-2, 3):
                for j in range(-2, 3):
                    mul = f_n_view[i + 2] * f_m_view[j + 2]
                    total += mul * c[s_n + i, s_m + j]
            y[n, m] = total


def spline_coeffs(s: np.ndarray, epsilon=1e-4, boundary_cond='edge', n_dims=2):
    """
    Determines the cubic B-spline coefficients of a signal. Uses the causal and
    anti-causal implementation. See Understanding Splines in your notes.


    Parameters
    ----------
    s: np.ndarray
        The signal which we determine the spline coefficients for.
    epsilon: float, optional
        The level of acceptable error associated with the first boundary
        condition
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.

    Returns
    -------
    c: np.ndarray
        The spline coefficients. Same dimensions as s.
    """

    c_plus = fwd_inv_filt(s, epsilon, n_dims=n_dims)
    c_minus = bwd_inv_filt(c_plus, n_dims=n_dims)
    if n_dims == 1:
        c = c_minus * 6
    elif n_dims == 2:
        c = c_minus * 36  # 6 squared because it's applied in both dims.
    return c


def fwd_inv_filt(x: np.ndarray, epsilon=1e-4, boundary_cond='edge', n_dims=2):
    """
    Implements the first causal part of the inverse filter. The transfer
    function is

    h(z) =           1
            -----------------
            1 - z_1 * z^{-1}
    where
    z_1 = -2 + sqrt(3)

    Boundary condition is:
    y[0] = sum_{k=0}^{k_0} x[k] z_1^k

    where k_0 > ln(epsilon)/ln(|z_1|)

    Parameters
    ----------
    x: np.ndarray
        The signal to be filtered using h(z), can be one or two dimensions.
    epsilon: float, optional
        The level of acceptable error associated with the first boundary
        condition
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.

    Returns
    -------
    y: np.ndarray
        The result of the filtering
    """

    cdef int k_0 = int(np.ceil(np.log(epsilon)) / np.log(np.abs(z_1)))

    pad_x = np.pad(x, k_0, mode=boundary_cond)

    y = np.empty_like(pad_x)
    tmp = np.empty_like(pad_x)
    # cdef double[:, :] tmp = tmp_arr
    # cdef double[:, :] y = y_arr
    cdef int N, M, idx
    if n_dims == 1:
        N = x.shape[0]
        z_1_powers = np.empty(k_0 + 1)
        for idx in range(k_0 + 1):
            z_1_powers[idx] = z_1 ** idx

        y[0] = np.sum(pad_x[0:k_0 + 1] * z_1_powers)
        for idx in range(1, N+k_0 * 2):
            y[idx] = y[idx - 1] * z_1 + pad_x[idx]
        return y[k_0: N+k_0]

    elif n_dims == 2:
        # Use tmp variable for the intermediate values when
        # using the seperability conditions...
        N = x.shape[0]
        M = x.shape[1]
        fwd_inv_filt2d(pad_x, tmp, y, k_0, N, M)
        return y[k_0: N+k_0, k_0: M+k_0]

    else:
        raise ValueError('Incompatible number of dimensions.')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void fwd_inv_filt2d(double[:, :] x, double[:,:] tmp, double[:, :] y,
                         int k_0, int N, int M):
    """
    Applies IIR filters in both directions and places the result in y.
    The transfer function is

    h(z) =           1
            -----------------
            1 - z_1 * z^{-1}
    where
    z_1 = -2 + sqrt(3)

    Boundary condition is:
    y[0] = sum_{k=0}^{k_0} x[k] z_1^k
    
    Parameters
    ----------
    x: double[:, :]
        array to be filtered. Assumed to be padded by the size of k_0
    tmp: double[:, :]
        tmp array
    y: double[:, :]
        result array
    k_0: int
        Amount of padding used to determine the boundary conditions
    N: int
        height of original array
    M: int
        width of original array

    Returns
    -------
    None
    """
    cdef double z_1_copy = z_1
    z_1_powers_arr = np.empty(k_0 + 1)
    cdef double[:] z_1_powers = z_1_powers_arr
    cdef int n, m, idx
    for idx in range(k_0 + 1):
        z_1_powers[idx] = z_1_copy ** idx

    cdef double s
    cdef int span_N = N + k_0 * 2
    cdef int span_M = M + k_0 * 2

    # horizontal first.
    for n in range(span_N):
        s = 0.
        for idx in range(k_0 + 1):
            s += x[n, idx] * z_1_powers[idx]
        tmp[n, 0] = s

        for m in range(1, span_M):
            tmp[n, m] = x[n, m] + z_1_copy * tmp[n, m - 1]

    # now vertical
    for m in range(span_M):
        # y[0, m] = np.sum(tmp[0:k_0 + 1, m] * z_1_powers)
        s = 0.
        for idx in range(k_0 + 1):
            s += tmp[idx, m] * z_1_powers[idx]
        y[0, m] = s

        for n in range(1, span_N):
            y[n, m] = tmp[n, m] + z_1_copy * y[n - 1, m]


def bwd_inv_filt(x: np.ndarray, epsilon=1e-4, boundary_cond='edge', n_dims=2):
    """
    Implements the second anti-causal part of the inverse filter. The transfer
    function is

    h(z) =        -z_1
            -----------------
            1 - z_1 * z
    where
    z_1 = -2 + sqrt(3)

    Boundary condition is:
    y[N-1] = z_1 * (x[N-1] + z_1 * x[N-1])
            ------------------------------
                1 - z_1 ^ 2

    Parameters
    ----------
    x: np.ndarray
        The signal to be filtered using h(z)
    boundary_cond: str (optional)
        as per np.pad(), typically 'edge' or reflect.

    Returns
    -------
    y: np.ndarray
        The result of the filtering
    """
    cdef int k_0 = int(np.ceil(np.log(epsilon)) / np.log(np.abs(z_1)))

    pad_x = np.pad(x, k_0, mode=boundary_cond)
    y = np.empty_like(pad_x)
    cdef int N, M, idx
    N = x.shape[0]

    if n_dims == 1:
        y[-1] = const * (pad_x[-1] + z_1 * pad_x[-2])
        for idx in range(k_0*2 + N-2, -1, -1):
            y[idx] = z_1 * (y[idx + 1] - pad_x[idx])
        return y[k_0:N+k_0]

    elif n_dims == 2:
        tmp = np.empty_like(pad_x)
        # Use tmp variable for the intermediate values when
        # using the seperability conditions...
        M = x.shape[1]
        bwd_inv_filt2d(pad_x, tmp, y, k_0, N, M)
        return y[k_0: N+k_0, k_0: M+k_0]

    else:
        raise ValueError('Incompatible number of definitions.')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void bwd_inv_filt2d(double[:, :] x, double[:, :] tmp, double[:, :] y,
                         int k_0, int N, int M):
    """
    Implements the second anti-causal part of the inverse filter. The transfer
    function is

    h(z) =        -z_1
            -----------------
            1 - z_1 * z
    where
    z_1 = -2 + sqrt(3)

    Boundary condition is:
    y[N-1] = z_1 * (x[N-1] + z_1 * x[N-1])
            ------------------------------
                1 - z_1 ^ 2
    Parameters
    ----------
    x
    tmp
    y
    k_0
    N
    M

    Returns
    -------

    """
    cdef double z_1_copy = z_1
    cdef double z_dash = const

    cdef int span_N = N-1 + k_0 * 2
    cdef int span_M = M-1 + k_0 * 2

    cdef int n, m
    # Horizontal
    for n in range(span_N, -1, -1):
        for m in range(span_M, -1, -1):
            if m == span_M:
                tmp[n, m] = z_dash * (x[n, m] + z_1_copy * x[n, m-1])
            else:
                tmp[n, m] = z_1_copy * (tmp[n, m+1] - x[n, m])
    # Vertical
    for m in range(span_M, -1, -1):
        for n in range(span_N, -1, -1):
            if n == span_N:
                y[n, m] = z_dash * (tmp[n, m] + z_1_copy * tmp[n-1, m])
            else:
                y[n, m] = z_1_copy * (y[n+1, m] - tmp[n, m])


def cubic_b_spline_1d(x_arr: np.ndarray):
    """
    We define the cubic b spline with

    for  0 <= |x| < 1
    y(x) = 2/3 - |x|^2 + |x|^3/2
    for  1 <= |x| < 2
    y(x) = (2 - |x|)^3/6
    elsewhere,
    y(x) = 0

    Parameters
    ----------
    x: np.ndarray
        contains the values to pass into the cubic b spline
    Returns
    -------
    y: np.ndarray
        the cubic b spline values
    """
    cdef int length = x_arr.size
    cdef np.ndarray[DTYPE_t, ndim=1] y_arr = np.empty_like(x_arr)
    cdef double[:] y = y_arr
    cdef double[:] x = x_arr
    _cubic_b_spline_1d(x, y, length)
    return y_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void _cubic_b_spline_1d(
        double[:] x,
        double[:] y,
        int length
):
    """
    We define the cubic b spline with

    for  0 <= |x| < 1
    y(x) = 2/3 - |x|^2 + |x|^3/2
    for  1 <= |x| < 2
    y(x) = (2 - |x|)^3/6
    elsewhere,
    y(x) = 0

    Parameters
    ----------
    x: np.ndarray
        contains the values to pass into the cubic b spline
    Returns
    -------
    y: np.ndarray
        the cubic b spline values
    """
    cdef int idx
    cdef double abs_x
    for idx in range(length):
        abs_x = fabs(x[idx])
        if abs_x < 1:
            y[idx] = 2./3. - abs_x**2 + abs_x**3 / 2
        elif abs_x < 2:
            y[idx] = (2. - abs_x)**3 / 6.
        else:
            y[idx] = 0


def cubic_b_spline(x: float):
    """
    We define the cubic b spline with

    for  0 <= |x| < 1
    y(x) = 2/3 - |x|^2 + |x|^3/2
    for  1 <= |x| < 2
    y(x) = (2 - |x|)^3/6
    elsewhere,
    y(x) = 0

    Parameters
    ----------
    x: float

    Returns
    -------
    y: float

    """
    abs_x = np.abs(x)
    if abs_x < 1:
        return 2.0/3.0 - abs_x ** 2 + abs_x ** 3 / 2
    elif np.abs(x) < 2:
        return (2 - abs_x) ** 3 / 6
    else:
        return 0


def derivative_cubic_spline1d(x_arr: np.ndarray):
    """
    The derivative of the cubic B spline is
    for -2 <= x < -1
    y'(x) = (x+2)^2/2
    for -1 <= x < 0
    y'(x) = -3x^2/2 - 2x
    for 0 <= x < 1
    y'(x) = 3x^2/2 - 2x
    for 1 <= x < 2
    y'(x) = -(x-2)^2/2

    Parameters
    ----------
    x_arr

    Returns
    -------

    """
    cdef int length = x_arr.size
    cdef double[:] x = x_arr
    cdef np.ndarray[DTYPE_t, ndim=1] y_arr = np.empty_like(x_arr)
    cdef double[:] y = y_arr
    _derivative_cubic_spline1d(x, y, length)
    return y_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void _derivative_cubic_spline1d(
        double[:] x,
        double[:] y,
        int length
):
    """
    The derivative of the cubic B spline is
    for -2 <= x < -1
    y'(x) = (x+2)^2/2
    for -1 <= x < 0
    y'(x) = -3x^2/2 - 2x
    for 0 <= x < 1
    y'(x) = 3x^2/2 - 2x
    for 1 <= x < 2
    y'(x) = -(x-2)^2/2

    Parameters
    ----------
    x

    Returns
    -------

    """
    cdef int idx

    cdef double x_val
    for idx in range(length):
        x_val = x[idx]
        if -1 > x_val >= -2.:
            y[idx] = (x_val + 2) ** 2 / 2.
        elif 0 > x_val >= -1:
            y[idx] = -3 * x_val ** 2 /2 - 2 * x_val
        elif 1 > x_val >= 0:
            y[idx] = 3 * x_val ** 2 /2 - 2 * x_val
        elif 2 > x_val >= 1:
            y[idx] = -(x_val - 2) ** 2 / 2
        else:
            y[idx] = 0.



