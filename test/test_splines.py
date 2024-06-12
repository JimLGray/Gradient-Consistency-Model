import splines as sp
import numpy as np
from scipy import ndimage
import simple


def test_interpolate_1D_impulse_no_shift():
    c = np.zeros(5)
    c[2] = 1.0
    y_ans = np.array(
        [0., 1., 4., 1., 0.]) / 6.  # This is a sampled cubic spline.
    v = np.zeros((5, 1))
    y = sp.interpolate(c, v, boundary_cond='reflect', n_dims=1)
    assert np.allclose(y_ans, y)


def test_interpolate_1D_impulse_unit_shift():
    c = np.zeros(5)
    c[2] = 1.0
    y_ans = np.array([1., 4., 1., 0., 1.]) / 6.
    v = np.ones((5, 1))
    y = sp.interpolate(c, v, boundary_cond='reflect', n_dims=1)
    assert np.allclose(y_ans, y)


def test_interpolate_1D_impulse_half_shift():
    c = np.zeros(10)
    c[2] = 1.0
    y_ans = np.zeros(10)
    # index zero is actually 2
    y_ans[0] = 1 / 6 * (2 - 1.5) ** 3
    y_ans[3] = y_ans[0]
    y_ans[1] = 2 / 3 - 0.5 ** 2 + 0.5 ** 3 / 2
    y_ans[2] = y_ans[1]
    v = np.ones((10, 1)) * 0.5
    y = sp.interpolate(c, v, n_dims=1)
    assert np.allclose(y_ans, y)


def test_interpolate_2D_impulse_no_shift():
    c = np.zeros((5, 5))
    c[2, 2] = 1.
    s_x = np.array([0., 1., 4., 1., 0.]) / 6.  # This is a sampled cubic spline.
    s_xx, s_yy = np.meshgrid(s_x, s_x)
    s_xy = s_xx * s_yy
    v = np.zeros((5, 5, 2))
    y = sp.interpolate(c, v)
    assert np.allclose(s_xy, y)


def test_interpolate_2D_quadruple_impulse_no_shift():
    s_x = np.array([0., 1., 4., 1., 0., 0., 0., 0., 1., 4., 1., 0.]) / 6.
    s_xx, s_yy = np.meshgrid(s_x, s_x)
    s_xy = s_xx * s_yy
    c = np.zeros_like(s_xy)
    c[2, 2] = 1.0
    c[2, 9] = 1.0
    c[9, 2] = 1.0
    c[9, 9] = 1.0
    v = np.zeros((12, 12, 2))
    y = sp.interpolate(c, v)

    assert np.allclose(s_xy, y)


def test_interpolate_2D_quadruple_half_half_shift():
    s_x = np.zeros(10)
    # index zero is actually 2
    s_x[0] = 1 / 6 * (2 - 1.5) ** 3
    s_x[3] = s_x[0]
    s_x[1] = 2 / 3 - 0.5 ** 2 + 0.5 ** 3 / 2
    s_x[2] = s_x[1]
    s_x = np.concatenate((s_x, s_x))
    s_xx, s_yy = np.meshgrid(s_x, s_x)
    s_xy = s_xx * s_yy

    c = np.zeros_like(s_xy)
    c[2, 2] = 1.0
    c[2, 12] = 1.0
    c[12, 2] = 1.0
    c[12, 12] = 1.0
    v = np.ones((20, 20, 2)) * 0.5
    y = sp.interpolate(c, v)
    assert np.allclose(s_xy, y)


def test_upscale_interpolate_impulse_2x():
    c = np.zeros((5, 5))
    c[2, 2] = 1.0
    ans_x = np.array(
        [0., 1. / 48., 1. / 6., 23. / 48., 4. / 6., 23. / 48., 1. / 6., 1. / 48.
            , 0., 0.])
    ans_xx, ans_yy = np.meshgrid(ans_x, ans_x)
    ans_xy = ans_xx * ans_yy
    res = np.empty_like((10, 10))
    v = np.zeros((10, 10, 2))
    res = sp.upscale_interpolate2d(c, v, 2)

    assert np.allclose(ans_xy, res)


def test_upscale_interpolate_impulse_2x_half_shift():
    c = np.zeros((5, 5))
    c[2, 2] = 1.0
    ans_x = np.array(
        [1. / 48., 1. / 6., 23. / 48., 4. / 6., 23. / 48., 1. / 6., 1. / 48.
            , 0., 0., 0.])
    ans_xx, ans_yy = np.meshgrid(ans_x, ans_x)
    ans_xy = ans_xx * ans_yy
    res = np.empty_like((10, 10))
    v = np.ones((10, 10, 2))
    res = sp.upscale_interpolate2d(c, v, 2)

    assert np.allclose(ans_xy, res)


def test_upscale_interpolate_impulse_4x_half_shift():
    c = np.zeros((5, 5))
    c[2, 2] = 1.0
    ans_x = np.array([
        1. / 384., 1. / 48., 9. / 128., 1. / 6., 121. / 384., 23. / 48.,
        235. / 384., 4. / 6., 235. / 384., 23. / 48., 121. / 384., 1. / 6.,
        9. / 128., 1. / 48., 1. / 384., 0., 0., 0., 0., 0.
    ])
    ans_xx, ans_yy = np.meshgrid(ans_x, ans_x)
    ans_xy = ans_xx * ans_yy
    v = np.ones((20, 20, 2))
    res = sp.upscale_interpolate2d(c, v, 4)
    assert np.allclose(ans_xy, res)



def test_spline_coeffs_1D_impulse():
    s = np.array([0, 0, 0, 0, 0, 0., 1., 4., 1., 0., 0, 0, 0, 0, 0]) / 6.
    c = sp.spline_coeffs(s, n_dims=1)
    c_ans = np.zeros_like(s)
    c_ans[7] = 1.0
    assert np.allclose(c_ans, c, atol=1e-3)


def test_spline_coeffs_2D_impulse():
    s_x = np.array([0, 0, 0, 0, 0, 0., 1., 4., 1., 0., 0, 0, 0, 0, 0]) / 6.
    s_xx, s_yy = np.meshgrid(s_x, s_x)
    s_xy = s_xx * s_yy
    c = sp.spline_coeffs(s_xy, n_dims=2)
    c_ans = np.zeros_like(s_xy)
    c_ans[7, 7] = 1.0
    assert np.allclose(c_ans, c, atol=1e-3)


def test_spline_coeffs_1D_double_impulse():
    s = np.array([0, 0, 1., 4., 1., 0, 0, 0., 0., 1., 4., 1., 0, 0, 0, 0, 0])
    s /= 6
    c = sp.spline_coeffs(s, n_dims=1)
    c_ans = np.zeros_like(s)
    c_ans[3] = 1.0
    c_ans[10] = 1.0
    assert np.allclose(c_ans, c, atol=1e-3)


def test_spline_coeffs_2D_quadruple_impulse():
    s_x = np.array([0, 0, 1., 4., 1., 0, 0, 0., 0., 1., 4., 1., 0, 0, 0, 0, 0])
    s_x /= 6
    s_xx, s_yy = np.meshgrid(s_x, s_x)
    s_xy = s_xx * s_yy
    c = sp.spline_coeffs(s_xy)
    c_ans = np.zeros_like(s_xy)
    c_ans[3, 3] = 1.0
    c_ans[3, 10] = 1.0
    c_ans[10, 3] = 1.0
    c_ans[10, 10] = 1.0
    assert np.allclose(c_ans, c, atol=1e-3)


def test_fwd_inv_filt_impulse1d():
    x = np.zeros(10)
    x[0] = 1.
    y = sp.fwd_inv_filt(x, boundary_cond='reflect', n_dims=1)
    y_ans = np.zeros_like(x)
    y_ans[0] = 1.
    for n in range(1, 10):
        y_ans[n] = y_ans[n - 1] * sp.z_1
    assert np.allclose(y_ans, y)


def test_bwd_inv_filt_impulse1d():
    x = np.zeros(5)
    x[4] = 1.
    y = sp.bwd_inv_filt(x, boundary_cond='reflect', n_dims=1)
    y_ans = np.zeros_like(x)
    y_ans[4] = x[4] * -sp.z_1
    for n in range(3, -1, -1):
        y_ans[n] = y[n + 1] * sp.z_1
    assert np.allclose(y_ans, y)


def test_fwd_inv_filt_impulse_2d():
    x = np.zeros((5, 5))
    x[0, 0] = 1.0
    y = sp.fwd_inv_filt(x, boundary_cond='reflect')
    y_ans = np.zeros_like(x)
    for n in range(5):
        for m in range(5):
            y_ans[n, m] = sp.z_1 ** (n + m)
    assert np.allclose(y_ans, y)


def test_bwd_inv_filt_impulse_2d():
    x = np.zeros((5, 5))
    x[4, 4] = 1.0
    y = sp.bwd_inv_filt(x, boundary_cond='reflect')
    y_ans = np.zeros_like(x)
    for n in range(5):
        for m in range(5):
            y_ans[n, m] = sp.z_1 ** ((4 - n) + (4 - m))
    y_ans *= sp.z_1 ** 2
    assert np.allclose(y_ans, y)


def test_cubic_b_spline_arr_ints():
    y_ans = np.array([0., 1., 4., 1., 0.]) / 6.
    x = np.arange(-2., 3.)
    y = sp.cubic_b_spline_1d(x)
    assert np.allclose(y_ans, y)


def test_cubic_b_spline_arr_halves():
    y_ans = np.zeros(5)
    # index zero is actually 2
    y_ans[0] = 1 / 6 * (2 - 1.5) ** 3
    y_ans[3] = y_ans[0]
    y_ans[1] = 2 / 3 - 0.5 ** 2 + 0.5 ** 3 / 2
    y_ans[2] = y_ans[1]
    x = np.arange(-2., 3.) + 0.5
    y = sp.cubic_b_spline_1d(x)
    assert np.allclose(y_ans, y)


def test_cubic_spline_zero():
    y_ans = 4. / 6.
    x = 0.0
    y = sp.cubic_b_spline(x)
    assert y == y_ans


def test_cubic_spline_half():
    y_ans = 2 / 3 - 0.5 ** 2 + 0.5 ** 3 / 2
    x = 0.5
    y = sp.cubic_b_spline(x)
    assert y == y_ans


def test_derivative_cubic_spline1d_ints():
    x = np.arange(-2., 3.)
    y_ans = np.zeros(5)
    # index zero is actually 2

    y_ans[0] = ((x[0] + 2) ** 2) / 2
    y_ans[1] = -x[1] ** 2 * 3 / 2 - x[1] * 2
    y_ans[2] = x[2] ** 2 * 3 / 2 - x[2] * 2
    y_ans[3] = -((x[3] - 2) ** 2) / 2
    y = sp.derivative_cubic_spline1d(x)
    assert np.allclose(y_ans, y)


def test_derivative_cubic_spline_1d_half():
    x = np.arange(-2., 3.) + 0.5
    y_ans = np.zeros(5)
    # index zero is actually 2

    y_ans[0] = (x[0] + 2) / 2 ** 2
    y_ans[1] = -x[1] ** 2 * 3 / 2 - x[1] * 2
    y_ans[2] = x[2] ** 2 * 3 / 2 - x[2] * 2
    y_ans[3] = -((x[3] - 2) ** 2) / 2

    y = sp.derivative_cubic_spline1d(x)
    assert np.allclose(y_ans, y)


def test_impulse_derivative_interpolate2d_y():
    c = np.zeros((11, 11))
    c[5, 5] = 1.0
    v = np.zeros((11, 11, 2))

    res = sp.upscale_interpolate2d(c, v, 1, order=(1, 0))
    ans = np.zeros((11, 11))
    ans[4:7, 4:7] = np.array([[1/12, 1/3, 1/12],
                             [0, 0, 0],
                             [-1/12, -1/3, -1/12]])
    assert np.allclose(res, ans)


def test_invertability_1d_impulse():
    s = np.zeros(11)
    s[5] = 1

    c = sp.spline_coeffs(s, n_dims=1)
    v = np.zeros((11, 2))
    res = sp.interpolate(c, v, n_dims=1)

    assert np.allclose(s, res, atol=1e-3)


def test_invertability_2d_impulse():
    s = np.zeros((11, 11))
    s[5, 5] = 1

    c = sp.spline_coeffs(s, n_dims=2)
    v = np.zeros((11, 11, 2))
    res = sp.interpolate(c, v, n_dims=2)

    assert np.allclose(s, res, atol=1e-3)


def test_invertability_2d_rand_padded():
    s = np.random.rand(11, 11)
    s_pad = np.pad(s, 1, mode='edge')
    c = sp.spline_coeffs(s_pad, n_dims=2)
    v = np.zeros((13, 13, 2))
    res_pad = sp.upscale_interpolate2d(c, v, scale=1, order=(0, 0))
    res = res_pad[1:-1, 1:-1]

    assert np.allclose(s, res, atol=1e-3)


def test_invertability_2d_grad_rand_padded():
    height = 15
    width = 15
    sig = 4/np.sqrt(2)
    r = np.random.rand(height, width)
    s = ndimage.gaussian_filter(r, sig, mode='nearest')
    s_pad = np.pad(s, 1, mode='edge')
    c = sp.spline_coeffs(s_pad, n_dims=2)

    v = np.zeros((height + 2, width + 2, 2))
    res_pad = sp.upscale_interpolate2d(c, v, scale=1, order=(1, 0))
    res = res_pad[1:-1, 1:-1]

    ans = ndimage.gaussian_filter(r, sigma=sig, order=(1, 0), mode='nearest')
    # Can't expect edges to be the same properly because boundary extension
    # isn't valid for gradients

    assert np.allclose(ans[3:-3, 3:-3], res[3:-3, 3:-3], atol=5e-3)


def test_invertability_2d_grad_horz_grad():
    height = 11
    width = 11
    sig = 1 / np.sqrt(2)
    r = np.ones((height, width))
    g = np.arange(1, height + 1)
    r *= g

    s = ndimage.gaussian_filter(r, sig, mode='nearest')
    s_pad = np.pad(s, 1, mode='edge')
    c = sp.spline_coeffs(s_pad, n_dims=2)

    v = np.zeros((height + 2, width + 2, 2))
    res_pad = sp.upscale_interpolate2d(c, v, scale=1, order=(0, 1))
    res = res_pad[1:-1, 1:-1]
    ans = ndimage.gaussian_filter(r, sig, order=(0, 1), mode='nearest')

    # Can't expect edges to be the same properly because boundary extension
    # isn't valid for gradients

    assert np.allclose(ans[3:-3, 3:-3], res[3:-3, 3:-3], atol=5e-3)


def test_invertability_2d_grad_stripes_grad():
    height = 11
    width = 11
    sig = 1 / np.sqrt(2)
    r = np.ones((height, width))
    g = np.mod(np.arange(1, height + 1), 2)
    r *= g

    s = ndimage.gaussian_filter(r, sig, mode='nearest')
    s_pad = np.pad(s, 1, mode='edge')
    c = sp.spline_coeffs(s_pad, n_dims=2)
    v = np.zeros((height + 2, width + 2, 2))
    res_pad = sp.upscale_interpolate2d(c, v, scale=1, order=(0, 1))
    res = res_pad[1:-1, 1:-1]
    ans = ndimage.gaussian_filter(r, sig, order=(0, 1), mode='nearest')
    # Can't expect edges to be the same properly because boundary extension
    # isn't valid for gradients

    assert np.allclose(ans[3:-3, 3:-3], res[3:-3, 3:-3], atol=5e-3)


def test_spline_downsampling_img_equivalence():
    height = 32
    width = 32
    q = 2
    sig = simple.get_sigma(q)
    r = np.random.rand(height, width)

    s = ndimage.gaussian_filter(r, sigma=sig, mode='nearest')
    s_down = s[::2 ** q, ::2 ** q]
    pad_down = np.pad(s_down, 1, mode='edge')
    c = sp.spline_coeffs(pad_down)

    d_height = s_down.shape[0]
    d_width = s_down.shape[1]
    v = np.zeros((d_height + 2, d_width + 2, 2))
    res = sp.upscale_interpolate2d(c, v, 1, (0, 0))[1:-1, 1:-1]

    assert np.allclose(s_down, res)


def test_spline_downsampling_first_grad_equivalence():
    height = 64
    width = 64
    q = 2
    sig = simple.get_sigma(q)
    r = np.random.rand(height, width)

    s = ndimage.gaussian_filter(r, sigma=sig, mode='nearest')
    s_down = s[::2 ** q, ::2 ** q]
    pad_down = np.pad(s_down, 1, mode='edge')
    c = sp.spline_coeffs(pad_down)

    d_height = s_down.shape[0]
    d_width = s_down.shape[1]
    v = np.zeros((d_height + 2, d_width + 2, 2))
    res = sp.upscale_interpolate2d(c, v, 1, (1, 0))[1:-1, 1:-1] / 2 ** q

    a = ndimage.gaussian_filter(r, sig, order=(1, 0), mode='nearest')
    ans = a[::2 ** q, ::2 ** q]

    assert np.allclose(ans, res)


def test_spline_downsampling_after_grad_equivalence():
    height = 64
    width = 64
    q = 2
    sig = simple.get_sigma(q)
    r = np.random.rand(height, width)

    s = ndimage.gaussian_filter(r, sigma=sig, mode='nearest')
    pad_s = np.pad(s, 1, mode='edge')
    c = sp.spline_coeffs(pad_s)

    v = np.zeros((height + 2, width + 2, 2))
    x = sp.upscale_interpolate2d(c, v, 1, (1, 0))[1:-1, 1:-1]
    res = x[::2 ** q, ::2 ** q]

    a = ndimage.gaussian_filter(r, sig, order=(1, 0), mode='nearest')
    ans = a[::2 ** q, ::2 ** q]

    assert np.allclose(ans, res)


def test_interpolate_smooth_impulse():
    height = 11
    width = 11
    q = 0
    sig = simple.get_sigma(q)

    c = np.zeros((height, width))
    v = np.zeros((height, width, 2))
    c[5, 5] = 1.0

    res = sp.interpolate_smooth(c, v, q)
    filt = np.array([0., 1., 4., 1., 0.]) / 6.
    tmp_x = ndimage.convolve1d(c, filt, axis=0, mode='nearest')
    tmp = ndimage.convolve1d(tmp_x, filt, axis=1, mode='nearest')

    ans = ndimage.gaussian_filter(tmp, sigma=sig, mode='nearest')
    assert np.allclose(ans, res)


def test_interpolate_smooth_rand():
    height = 11
    width = 11
    q = 0
    sig = simple.get_sigma(q)

    c = np.random.random((height, width))
    v = np.zeros((height, width, 2))
    c[5, 5] = 1.0
    filt_ext = 2

    pad_c = np.pad(c, filt_ext, mode='edge')

    res = sp.interpolate_smooth(c, v, q)
    filt = np.array([0., 1., 4., 1., 0.]) / 6.
    tmp_x = ndimage.convolve1d(pad_c, filt, axis=0, mode='nearest')
    tmp = ndimage.convolve1d(tmp_x, filt, axis=1, mode='nearest')
    unpad = tmp[filt_ext:-filt_ext, filt_ext:-filt_ext]

    ans = ndimage.gaussian_filter(unpad, sigma=sig, mode='nearest')
    assert np.allclose(ans, res)


def test_interpolate_smooth_equivalence_impulse():
    height = 11
    width = 11
    q = 0
    sig = simple.get_sigma(q)
    r = np.zeros((height, width))
    r[5, 5] = 1.0
    pad_r = np.pad(r, 2, mode='edge')

    s = ndimage.gaussian_filter(r, sigma=sig, mode='nearest')
    c = sp.spline_coeffs(pad_r)

    v = np.zeros((height + 4, width + 4, 2))
    res = sp.interpolate_smooth(c, v, q, (0, 0))[2:-2, 2:-2]
    diff = res - s

    assert np.allclose(s, res, atol=1e-4)


def test_interpolate_smooth_equivalence_rand():
    height = 11
    width = 11
    q = 2
    pad_width = 2 ** (q+1)
    sig = simple.get_sigma(q)
    r = np.random.rand(height, width)
    pad_r = np.pad(r, pad_width, mode='edge')

    c = sp.spline_coeffs(pad_r)

    v = np.zeros((height + 2 * pad_width, width + pad_width * 2, 2))
    pad_res = sp.interpolate_smooth(c, v, q, (0, 0))
    res = pad_res[pad_width:-pad_width, pad_width:-pad_width]

    s = ndimage.gaussian_filter(r, sigma=sig, mode='nearest')
    diff = res - s
    assert np.allclose(s, res, atol=1e-4)


def test_interpolate_smooth_grad_equivalence():
    height = 11
    width = 11
    q = 0
    r = np.ones((height, width))
    g = np.arange(1, height + 1)
    r *= g
    sig = simple.get_sigma(q)
    pad_width = 2 ** (q + 1)

    pad_r = np.pad(r, pad_width, mode='edge')
    c = sp.spline_coeffs(pad_r)

    v = np.zeros((height + pad_width * 2, width + pad_width * 2, 2))
    x = sp.interpolate_smooth(c, v, q, (0, 1))
    res = x[pad_width:-pad_width, pad_width:-pad_width]

    ans = ndimage.gaussian_filter(r, sig, order=(0, 1), mode='nearest')

    assert np.allclose(ans, res)
