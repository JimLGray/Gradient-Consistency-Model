import numpy as np
from data_io import fileIO, plot
from scipy import ndimage
from simple import get_sigma
from matplotlib import pyplot as plt
import splines as sp

"""
The goal of this is to determine if our approach makes sense and if we're doing 
the spline gradients in a sensible way.

The plan is to do everything in 1D with q = 3, and see what's going on. 
"""

# Let's start with a row of an image and then downsampling it.


def plot1d(data, title):
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Intensity')
    plt.show()


def hanning_win(x_axis):
    M = x_axis.size
    win = 0.5 + 0.5 * np.cos(2 * np.pi * x_axis / (M-1))
    return win


def one_dimensional_test():
    img = fileIO.open_img('data/additional/boardgames/input_Cam040.png')
    row = img[55, 0:128]
    # row = np.zeros(128)
    # row[63] = 1.0

    q = 2
    scale = 2 ** q
    sigma = get_sigma(q)
    h = 30
    filt_row = ndimage.gaussian_filter(row, sigma)

    dog_res = ndimage.gaussian_filter(row, sigma, order=1)

    # plot1d(row, 'Row')
    plot1d(filt_row, 'Filtered Row')
    plot1d(dog_res, 'Derivative of Gaussians (Ideal)')

    grad_filt = np.array([1, 0, -1]) / 2.0

    diff_gauss = ndimage.convolve(filt_row, grad_filt)
    plot1d(diff_gauss, 'Difference of Gaussians')

    down_row = filt_row[::scale]
    # plot1d(down_row, 'Downsampled Row')

    diff_gauss_coarse = ndimage.convolve(down_row, grad_filt) / scale
    plot1d(diff_gauss_coarse, 'Difference of Gaussians Coarse Scale')

    spl_fine = sp.spline_coeffs(filt_row, n_dims=1)
    spl_coarse = sp.spline_coeffs(down_row, n_dims=1)

    # plot1d(spl_fine, 'Spline Coefficients Fine Scale')
    # plot1d(spl_coarse, 'Spline Coefficients Coarse Scale')
    # also need to do an upscaling operation too.

    fine_spline_grad = sp.interpolate(spl_fine, np.zeros((128, 1)), (1, 0), 'edge',
                                      1)
    plot1d(fine_spline_grad, 'Spline Gradient Fine Scale')
    # this one is good. Now to check at the coarse scale too.
    coarse_spline_grad = sp.interpolate(spl_coarse, np.zeros((128, 1)), (1, 0),
                                        'edge', 1) / scale
    plot1d(coarse_spline_grad, 'Spline Gradient Coarse Scale')
    upscaled_coarse_grad = sp.upscale_interpolate1d(spl_coarse, np.zeros((128, 1)),
                                                    scale, order=1,
                                                    boundary_cond='edge') / scale
    plot1d(upscaled_coarse_grad, 'Upscaled Spline Gradient Coarse Scale')


def two_dimensional_test():
    img = fileIO.open_img('data/additional/boardgames/input_Cam040.png')
    patch = img[0:128, 0:128]
    # row = np.zeros(128)
    # row[63] = 1.0
    # plot.colorbar_img_plot(patch, title='Raw Image')

    q = 2
    scale = 2 ** q
    sigma = get_sigma(q)
    h = 30

    filt_patch = ndimage.gaussian_filter(patch, sigma, mode='nearest')
    down_patch = filt_patch[::4, ::4]

    plot.colorbar_img_plot(filt_patch, title='Gaussian Smoothed Image')
    patch_x = ndimage.gaussian_filter(patch, sigma, (0, 1), mode='nearest')
    patch_y = ndimage.gaussian_filter(patch, sigma, (1, 0), mode='nearest')
    plot.colorbar_img_plot(patch_x[:, 127:128], title='X direction Derivative of Gaussians')
    # plot.colorbar_img_plot(patch_y, title='Y direction Derivative of Gaussians')

    sp_coeffs = sp.spline_coeffs(filt_patch)
    v = np.zeros((128, 128, 2))
    v_small = np.zeros((32, 32, 2))
    # plot.colorbar_img_plot(sp_coeffs, title='Spline Coefficients')
    sp_x = sp.interpolate(sp_coeffs, v, (0, 1))
    sp_y = sp.interpolate(sp_coeffs, v, (1, 0))
    plot.colorbar_img_plot(sp_x[:, 127:128], 'Spline Gradient X direction Top Scale')
    # plot.colorbar_img_plot(sp_y, 'Spline Gradient Y direction Top Scale')

    sp_coarse = sp.spline_coeffs(down_patch)
    sp_coarse_x = sp.upscale_interpolate2d(sp_coarse, v, scale, (0, 1)) / scale
    # sp_coarse_y = sp.upscale_interpolate2d(sp_coarse, v, scale, (1, 0))
    plot.colorbar_img_plot(sp_coarse_x[:, 127:128], 'Spline Gradient X direction Upscaled Coarse Scale')
    # plot.colorbar_img_plot(sp_coarse_y, 'Spline Gradient Y direction Upscaled Coarse Scale')


    # sp_coarse_x = sp.upscale_interpolate2d(sp_coarse, v_small, 1, (0, 1))
    # sp_coarse_y = sp.upscale_interpolate2d(sp_coarse, v_small, 1, (1, 0))
    # plot.colorbar_img_plot(sp_coarse_x, 'Spline Gradient X direction Coarse Scale')
    # plot.colorbar_img_plot(sp_coarse_y, 'Spline Gradient Y direction Coarse Scale')

two_dimensional_test()