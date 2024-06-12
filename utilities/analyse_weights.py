from data_io import plot
import numpy as np
from scipy import ndimage

w_name = 'output/scratch/additional/antinous/gc_model/alpha=0.2/Weights_0_49.npy'
# reg_name = 'output/scratch/antinous/gc_model/alpha=0.2/reg_graph_0_1.npy'
# weighted_grad_name = 'output/scratch/antinous/gc_model/alpha=0.2/Weighted_Grads_0_1.npy'
# weighted_diff_name = 'output/scratch/antinous/gc_model/alpha=0.2/Weighted_Diffs_0_1.npy'
# diff_name = 'output/scratch/antinous/gc_model/alpha=0.2/Diffs_Unweighted_0_1.npy'
# grad_name = 'output/scratch/antinous/gc_model/alpha=0.2/Grads_Unweighted_0_1.npy'

# context_name = 'output/scratch/antinous/gc_model/alpha=0.2/context_0.json'
# plt_baselines = [
#     (0, 1, 0)
#     (4, 0, 0), (4, 0, 1), (4, 0, 2),
#     (0, -4, 0), (0, -4, 1), (0, -4, 2)
# ]

# plot.basic_weights_plotter(w_name, context_name, plt_baselines, normalise=True)

# window = (0, 25, 175, 200)
# window = (50, 100, 50, 100)
# window = (75, 125, 325, 375)
window = (0, 512, 0, 512)

view = None
plot.full_weights_plotter(w_name, window=window, view=view)
# plot.full_weights_plotter(weighted_grad_name, window=window, view=view)
# plot.full_weights_plotter(weighted_diff_name, window=window, view=view)
# plot.full_weights_plotter(diff_name, window=window, view=view)
# plot.full_weights_plotter(grad_name, window=window, view=view)

# plot.full_reg_weights_plotter(reg_name, 1, window)
# plot.reg_weights_plotter(reg_name, 1.0, window)

# disp_name = 'output/scratch/antinous/gc_model/alpha=0.2/Disparity_0_1.npy'
# disparity = np.load(disp_name)
# old_disp = np.load('output/scratch/antinous/gc_model/alpha=0.2/Disparity_0_1.npy')
# #
# plot.colorbar_img_plot(disparity[window[2]:window[3], window[0]:window[1]],
#                        title='Disparity'
#                        )
# grad_operator = np.array([1.0, 0.0, -1.0]) / 2.0
# ave_operator = np.array([1.0, 1.0])/2.0
# w_x = ndimage.convolve1d(old_disp, grad_operator, axis=1, mode='constant')
# w_y = ndimage.convolve1d(old_disp, grad_operator, axis=0, mode='constant')
# reciprocal_mag = 1 / (2 * np.sqrt(w_x ** 2 + w_y ** 2 + 1e-7))
# reg_weights_ans = ndimage.convolve1d(reciprocal_mag, ave_operator, axis=0,
#                                      mode='constant') * 0.0005
# plot.colorbar_img_plot(reg_weights_ans[window[2]:window[3], window[0]:window[1]])

# sum_gpg = np.load('output/scratch/vinyl/gc_model/alpha=0.2_l1_l1/sum_gpq_0_1.npy')
# plot.colorbar_img_plot(sum_gpg[window[2]:window[3], window[0]:window[1]],
#                        title='sum_gpq')
# delta_ipq = np.load('output/scratch/vinyl/gc_model/alpha=0.2_l1_l1/sum_delta_ipq_0_1.npy')
# plot.colorbar_img_plot(delta_ipq[window[2]:window[3], window[0]:window[1]],
#                        title='sum_delta_ipq')