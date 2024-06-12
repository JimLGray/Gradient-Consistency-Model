import numpy as np
from scipy import io, signal
from data_io import plot, fileIO
import h5py

res_dir = 'output/blender/buddha/reference/'
f = h5py.File('data/blender/buddha/lf.h5')
dH = f.attrs['dH'][0]
focal_length = f.attrs['focalLength'][0]
delta_x = f.attrs['shift']
gt_depth_name = 'data/blender/buddha/imgs/gt_depth_lowres.pfm'
gt_depth, _ = fileIO.readPFM(gt_depth_name)

gt_name = 'data/blender/buddha/imgs/gt_disp_lowres.pfm'
gt_disp, _ = fileIO.readPFM(gt_name)
# plot.colorbar_img_plot(gt_disp)

bad_px = np.full(3, np.NaN)
rmse = np.full(3, np.NaN)
threshold = 0.2

for i in range(2):
    disparity = io.loadmat(res_dir + 'Disparity_' + str(i) + '.mat')['save_z']
    np.save(res_dir + 'Disparity_' + str(i) + '.npy', disparity)
    plot.colorbar_img_plot(disparity, save=True, title='Disparity',
                           filename=res_dir + 'Disparity_' + str(i) + '.png')
    if disparity.shape != gt_disp.shape:
        w = signal.resample_poly(disparity, gt_disp.shape[1],
                                 disparity.shape[1],
                                 axis=1, padtype='edge')
        w = signal.resample_poly(w, gt_disp.shape[0], w.shape[0], axis=0,
                                 padtype='edge')
    else:
        w = disparity

    rmse[i] = np.sqrt(np.nanmean((w - gt_disp) ** 2))
    num = dH * focal_length
    den = w + delta_x
    depth = num / den
    bad_px[i] = plot.bad_px_percent(threshold, depth, gt_depth) * 100

print(bad_px)
print(rmse)
