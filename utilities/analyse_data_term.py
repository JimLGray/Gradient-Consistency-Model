import numpy as np
from data_io import plot
from scipy import  io

our_data_0 = np.load('output/blender/buddha/ctf_true/data_term0.npz')
their_data_0 = io.loadmat('output/blender/buddha/reference/data_term0.mat')

our_J11 = our_data_0['J11']
their_J11 = their_data_0['J11']

diff_J11 = np.abs(our_J11 - their_J11)
plot.colorbar_img_plot(diff_J11, title='J11, 0th Warp')

our_J12 = -our_data_0['J12']
their_J12 = their_data_0['J12']
diff_J12 = np.abs(our_J12 - their_J12)
# I think ours has to be multpiplied by -1 because we change the coordinate axes
# and multiply their disparity by -1
plot.colorbar_img_plot(diff_J12, title='J12, 0th Warp')

diff_J22 = np.abs(our_data_0['J22'] - their_data_0['J22'])
plot.colorbar_img_plot(diff_J22,  title='J22, 0th Warp')
