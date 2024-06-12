import numpy as np
from data_io import plot

ours_0 = np.load('output/blender/buddha/ctf_true/Disparity_0_24.npy')
theirs_0 = np.load('output/blender/buddha/reference/Disparity_0.npy')

diff_0 = np.abs(ours_0 - theirs_0)

plot.colorbar_img_plot(diff_0, 'Difference, Warp 0')

ours_1 = np.load('output/blender/buddha/ctf_true/Disparity_1_24.npy')
theirs_1 = np.load('output/blender/buddha/reference/Disparity_1.npy')

diff_1 = np.abs(ours_1 - theirs_1)

plot.colorbar_img_plot(diff_1, 'Difference, Warp 1')