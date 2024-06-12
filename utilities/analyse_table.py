import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('output/training/table.csv')
scene = 'sideboard'

scene_data = df[df['Scene'] == scene]
methods = ['old', 'warp_free', 'inter_irls', 'grad_const']
alphas = [0.0005, 0.00025, 0.00025, 0.0005]
method_alpha_pairs = zip(methods, alphas)

metric = 'Photometric Error'
# metric = 'RMSE'

for method, alpha in method_alpha_pairs:
    cond = (scene_data['Method'] == method) & (scene_data['Reg_const'] == alpha)
    method_data = scene_data[cond]
    plt.plot(method_data['Ax=b Solves'], method_data[metric])

plt.title(metric + ' vs. Ax=b Solves in ' + scene)
plt.legend([
    'Traditional Disparity Estimation',
    'Warping Free',
    'Interpolation in IRLS and Warping Free',
    'Gradient Consistency with Warp Free and Interpolation in IRLS',
])
plt.xlabel('Ax=b Solves')
plt.ylabel(metric)
plt.show()
