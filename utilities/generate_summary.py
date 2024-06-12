import numpy as np
import pandas as pd
import itertools

table = pd.read_csv('output/cse_sum/50_solves/additional/table.csv')
lines = table.shape[0]

scenes = table['Scene'].unique()
methods = list(table['Method'].unique())
reg_consts = table['Reg_const'].unique()

metrics = [
    'RMSE',
    'Photometric Error',
    'Ax=b Solves'
]
mm_pairs = list(itertools.product(methods, metrics))
res_cols = []
for pair in mm_pairs:
    col_str = pair[0] + ' ' + pair[1]
    res_cols.append(col_str)

cols = ['Scene', 'Reg_const'] + res_cols
print(cols)

data = np.full((len(scenes) * len(reg_consts), len(cols)), np.nan)

summary_table = pd.DataFrame(data, columns=cols)

idx = 0
for scene in scenes:
    scene_table = table[table['Scene'] == scene]
    # print(scene_table)
    for reg_const in reg_consts:
        reg_table = scene_table[scene_table['Reg_const'] == reg_const]
        rmse_solves = []
        for method in methods:
            method_table = reg_table[reg_table['Method'] == method]
            res_row = method_table.tail(1)
            # print(res_row.shape)
            if res_row.shape[0] == 0:
                rmse = np.nan
                photo_err = np.nan
                solves = np.nan
            else:
                rmse = res_row['RMSE'].iloc[0]
                photo_err = res_row['Photometric Error'].iloc[0]
                solves = res_row['Ax=b Solves'].iloc[0]
            # print(rmse, solves)
            rmse_solves.append(rmse)
            rmse_solves.append(photo_err)
            rmse_solves.append(solves)
        summary_table.iloc[idx] = [scene, reg_const] + rmse_solves

        idx += 1

reg_table = summary_table.groupby(['Reg_const']).mean(numeric_only=True)
new_index = np.arange(idx, idx + len(reg_consts), dtype=float)
reg_table = reg_table.reset_index(names='Reg_const')
reg_table.insert(0, 'Scene', 'Average')

summary_table = pd.concat([summary_table, reg_table], ignore_index=True)

summary_table.to_csv('output/cse_sum/50_solves/additional/summary.csv')
# Need an average row for
