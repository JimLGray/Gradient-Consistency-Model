import simple
import numpy as np
from scipy import signal, ndimage
import pytest


def test_trunc_gaussian1d_impulse_dir_0_ax_0():
    sigma = 2.0/3.0
    size = 5
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    g = ndimage.gaussian_filter1d(impulse, sigma)
    t_g = np.zeros_like(g)
    t_g[0:size+1] = g[0:size+1]
    t_g = t_g / np.sum(t_g)
    img = np.zeros(((2 * size + 1), (2 * size + 1)))
    img[5, 5] = 1.0

    ans = np.copy(img)
    ans[:, size] = t_g

    res = simple.trunc_gaussian1d(img, 0, 0, sigma, 'nearest', 0.0)
    assert np.allclose(res, ans)


def test_trunc_gaussian1d_impulse_dir_0_ax_1():
    sigma = 2.0/3.0
    size = 5
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    g = ndimage.gaussian_filter1d(impulse, sigma)
    t_g = np.zeros_like(g)
    t_g[0:size+1] = g[0:size+1]
    t_g = t_g / np.sum(t_g)
    img = np.zeros(((2 * size + 1), (2 * size + 1)))
    img[5, 5] = 1.0

    ans = np.copy(img)
    ans[size, :] = t_g

    res = simple.trunc_gaussian1d(img, 1, 0, sigma, 'nearest', 0.0)
    assert np.allclose(res, ans)


def test_trunc_gaussian1d_impulse_dir_1_ax_0():
    sigma = 2.0/3.0
    size = 5
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    g = ndimage.gaussian_filter1d(impulse, sigma)
    t_g = np.zeros_like(g)
    t_g[size:] = g[size:]
    t_g = t_g / np.sum(t_g)
    img = np.zeros(((2 * size + 1), (2 * size + 1)))
    img[5, 5] = 1.0

    ans = np.copy(img)
    ans[:, size] = t_g

    res = simple.trunc_gaussian1d(img, 0, 1, sigma, 'nearest', 0.0)
    assert np.allclose(res, ans)


def test_trunc_gaussian1d_impulse_dir_1_ax_1():
    sigma = 2.0/3.0
    size = 5
    impulse = np.zeros((2 * size + 1))
    impulse[size] = 1.0
    g = ndimage.gaussian_filter1d(impulse, sigma)
    t_g = np.zeros_like(g)
    t_g[size:] = g[size:]
    t_g = t_g / np.sum(t_g)
    img = np.zeros(((2 * size + 1), (2 * size + 1)))
    img[5, 5] = 1.0

    ans = np.copy(img)
    ans[size, :] = t_g

    res = simple.trunc_gaussian1d(img, 1, 1, sigma, 'nearest', 0.0)
    assert np.allclose(res, ans)


def test_filter1d_invalid_even():
    kernel = np.array([1, 1], dtype=np.float64)
    img = np.random.random((100, 100))
    axis = 0
    mode = 'constant'
    with pytest.raises(ValueError):
        simple.debug_filter1d(img, kernel, axis, mode)


# def test_filter1d_invalid_mode():
#     kernel = np.array([1, 1, 1], dtype=np.float64)
#     img = np.random.random((100, 100))
#     axis = 0
#     mode = 'kek'
#     with pytest.raises(ValueError):
#         basic.debug_filter1d(img, kernel, axis, mode)


def test_filter1d_invalid_axis():
    kernel = np.array([1, 1, 1], dtype=np.float64)
    img = np.random.random((100, 100))
    axis = 5
    mode = 'constant'
    with pytest.raises(ValueError):
        simple.debug_filter1d(img, kernel, axis, mode)



def test_filter1d_rand_constant():
    kernel = np.random.random(11)
    img = np.random.random((100, 120))
    axis = 0
    mode = 'constant'
    result = simple.debug_filter1d(img, kernel, axis, mode)
    ans = ndimage.convolve1d(img, kernel, axis=axis, mode=mode)
    assert np.allclose(ans, result)


def test_filter1d_rand_reflect():
    kernel = np.random.random(11)
    img = np.random.random((100, 120))
    axis = 0
    mode = 'reflect'
    result = simple.debug_filter1d(img, kernel, axis, mode)
    ans = ndimage.convolve1d(img, kernel, axis=axis, mode='mirror')
    assert np.allclose(ans, result)


def test_filter2d_sep_invalid_even():
    kernel = np.array([1, 1], dtype=np.float64)
    img = np.random.random((100, 100))
    mode = 'constant'
    with pytest.raises(ValueError):
        simple.debug_filt_sep2d(img, kernel, mode)


# def test_filter2d_invalid_mode():
#     kernel = np.array([1, 1, 1], dtype=np.float64)
#     img = np.random.random((100, 100))
#     mode = 'kek'
#     with pytest.raises(ValueError):
#         basic.debug_filt_sep2d(img, kernel, mode)


def test_filter2d_sep_rand_reflect():
    kernel = np.random.random(3)
    img = np.random.random((5, 6))
    mode = 'reflect'
    result = simple.debug_filt_sep2d(img, kernel, mode)
    ans = signal.sepfir2d(img, kernel, kernel)
    assert np.allclose(ans, result)


def test_filter2d_fill():
    img = np.random.random((100, 100))
    boundary = 'fill'
    mode = 'constant'
    kernel = np.random.random((5, 5))
    ans = signal.convolve2d(img, kernel, mode='same', boundary=boundary)
    res = simple.debug_filter2d(img, kernel, mode)
    assert np.allclose(ans, res)


def test_filter2d_symmetric():
    img = np.random.random((100, 100))
    boundary = 'symm'
    mode='symmetric'
    kernel = np.random.random((5, 5))
    ans = signal.convolve2d(img, kernel, mode='same', boundary=boundary)
    res = simple.debug_filter2d(img, kernel, mode)
    assert np.allclose(ans, res)


def test_upsample_img():
    img = np.random.random((100, 100))
    output = simple.resample_img(img, 2, 1)
    ans = signal.resample_poly(img, 2, 1, axis=0, padtype='antireflect')
    ans = signal.resample_poly(ans, 2, 1, axis=1, padtype='antireflect')
    assert np.allclose(ans, output)


def test_downsample_img():
    img = np.random.random((100, 100))
    output = simple.resample_img(img, 1, 2)
    ans = signal.resample_poly(img, 1, 2, axis=0, padtype='antireflect')
    ans = signal.resample_poly(ans, 1, 2, axis=1, padtype='antireflect')
    assert np.allclose(ans, output)


def test_pred_err_no_err():
    ref = np.random.random((100, 100))
    tgt = np.copy(ref)
    disparity = np.zeros_like(ref)
    baseline = (1, 0)

    res = simple.pred_err(disparity, baseline, ref, tgt)
    ans = 0

    assert res == ans


def test_pred_err_one():
    ref = np.random.random((100, 100))
    tgt = np.copy(ref) + 1.0
    disparity = np.zeros_like(ref)
    baseline = (1, 0)

    res = simple.pred_err(disparity, baseline, ref, tgt)
    ans = 1.0

    assert res == ans


def test_tuple_search_not_present():
    a = np.arange(10)
    a = a.reshape(5, 2)
    t = (-5, -5)

    res = simple.tuple_search(a, t)
    ans = -1

    assert res == ans


def test_tuple_search_first():
    a = np.arange(10)
    a = a.reshape(5, 2)
    t = (0, 1)

    res = simple.tuple_search(a, t)
    ans = 0

    assert res == ans


def test_tuple_search_middle():
    a = np.arange(10)
    a = a.reshape(5, 2)
    t = (4, 5)

    res = simple.tuple_search(a, t)
    ans = 2

    assert res == ans


def test_tuple_search_end():
    a = np.arange(10)
    a = a.reshape(5, 2)
    t = (8, 9)

    res = simple.tuple_search(a, t)
    ans = 4

    assert res == ans


def test_weighted_sad_rand_weights_zero_img():
    w_height = 3
    w_width = 7
    height = 21
    width = 15
    x = np.zeros((height, width))
    weights = np.random.random((w_height, w_width))
    y = simple.weighted_sad(x, weights)
    ans = np.zeros_like(x)

    assert np.allclose(y, ans)


def test_weighted_sad_ones_weights_impulse():
    w_height = 3
    w_width = 3
    height = 11
    width = 11

    x = np.zeros((height, width))
    x[5, 5] = 1.0

    weights = np.ones((w_height, w_width))
    y = simple.weighted_sad(x, weights)
    ans = np.zeros_like(x)
    ans[4:7, 4:7] = [[1, 1, 1], [1, 8, 1], [1, 1, 1]]

    assert np.allclose(y, ans)


def test_weighted_rand_weights_impulse():
    w_height = 3
    w_width = 3
    height = 11
    width = 11

    x = np.zeros((height, width))
    x[5, 5] = 1.0
    weights = np.random.random((w_height, w_width))
    weights[1, 1] = 0
    y = simple.weighted_sad(x, weights)
    ans = np.zeros_like(x)
    ans[4:7, 4:7] = np.flip(weights)
    ans[5, 5] = np.sum(weights)
    assert np.allclose(y, ans)


def test_weighted_rand_weights_negative_impulse():
    w_height = 3
    w_width = 3
    height = 11
    width = 11

    x = np.zeros((height, width))
    x[5, 5] = -1.0
    weights = np.random.random((w_height, w_width))
    weights[1, 1] = 0
    y = simple.weighted_sad(x, weights)
    ans = np.zeros_like(x)
    ans[4:7, 4:7] = np.flip(weights)
    ans[5, 5] = np.sum(weights)
    assert np.allclose(y, ans)
