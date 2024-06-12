'''
Performs 5 Fold Cross Validation on combined.csv
'''

import pandas as pd
import numpy as np
import itertools
import copy

table = pd.read_csv('output/combined_rmse.csv')

scenes = list(table['Scene'].unique())
reg_consts = np.array([0.0005, 0.001, 0.002, 0.005])

n_scenes = len(scenes)
n_blocks = 5
block_len = int(n_scenes / n_blocks)
print(block_len)

test_sets = []
train_sets = []

# Divide Data into the 5 folds/blocks
for block in range(n_blocks):
    test_set = []
    train_set = copy.deepcopy(scenes)
    for b_idx in range(block_len):
        idx = b_idx + block * block_len
        scene = scenes[idx]
        test_set.append(scene)
        train_set.remove(scene)
    test_sets.append(test_set)
    train_sets.append(train_set)


def ave_scenes(
        df: pd.DataFrame,
        scene_list: list,
        alpha_vals: list,
        label: str,
        fold: int
):
    """
    Average across scenes in scene list for a given set of alpha values.

    Parameters
    ----------
    df: pd.Dataframe
    scene_list: list
    alpha_vals: list
        list of alpha values
    label: str
        labels the average set
    fold: int
        which fold

    Returns
    -------

    """
    tab = df[df['Scene'].isin(scene_list)]
    tab = tab[tab['Reg_const'].isin(alpha_vals)]
    ave = tab.groupby('Reg_const').mean(numeric_only=True)
    ave = ave[ave.columns[1:]]
    ave.reset_index(names='Reg_const', inplace=True)
    ave.insert(0, 'Label', label)
    ave.insert(1, 'Fold', fold)
    return ave


data = np.empty((2 * len(reg_consts) * n_blocks, len(table.columns)))
col_list = ['Label', 'Fold'] + list(table.columns[2:])
out = pd.DataFrame(data, columns=col_list)

for idx in range(n_blocks):
    test_set = test_sets[idx]
    train_set = train_sets[idx]

    test_ave = ave_scenes(table, test_set, reg_consts, 'Test', idx)
    train_ave = ave_scenes(table, train_set, reg_consts, 'Train', idx)
    out_idx = idx * 2 * len(reg_consts)
    out.iloc[out_idx: out_idx+len(reg_consts)] = test_ave.values
    out.iloc[out_idx+len(reg_consts): out_idx+2*len(reg_consts)] = \
        train_ave.values

pruned_cols = []
for col in col_list:
    if col.endswith('Ax=b Solves'):
        pruned_cols.append(col)

out.drop(columns=pruned_cols, inplace=True)

# Now to figure out the actual result of the 5 fold validation.

# need table of regulariser constants
reg_table = np.empty((n_blocks, ))

col_list = out.columns

pruned_cols = []
for col in col_list:
    if col.endswith('Photometric Error'):
        pruned_cols.append(col)
pruned_cols += ['Label', 'Fold', 'Reg_const']

# List of columns, one for each method
remaining_cols = []
for col in col_list:
    if col not in pruned_cols:
        remaining_cols.append(col)

# list of columns that have the best of results.
best_cols = []
for col in col_list:
    if col not in ['Label', 'Fold', 'Reg_const']:
        best_cols.append(col)


# columns that the final table will have
final_cols = []
for c in range(len(remaining_cols)):
    best_col_rmse = best_cols[c * 2]
    best_col_photo = best_cols[c * 2 + 1]
    reg_col_name = remaining_cols[c][0:-4] + ' alpha'
    final_cols.append(best_col_rmse)
    final_cols.append(best_col_photo)
    final_cols.append(reg_col_name)

best_test = pd.DataFrame(
    np.empty((n_blocks, len(remaining_cols) * 3)),
    columns=final_cols
)

for block in range(n_blocks):
    fold_tab = out[out['Fold'] == block]
    train_res = fold_tab[fold_tab['Label'] == 'Train']
    prune_train = train_res.drop(columns=pruned_cols)
    min_idx = prune_train.idxmin()
    test_choice = min_idx - block_len
    best_vals = []
    for col in final_cols:
        if col in remaining_cols:
            idx = test_choice[col]
            best_rmse = out[col].iloc[idx]
            best_vals.append(best_rmse)
            alpha = out['Reg_const'].iloc[idx]
        elif col in best_cols:
            best_photo = out[col].iloc[idx]
            best_vals.append(best_photo)
        else:
            best_vals.append(alpha)

    best_test.iloc[block] = best_vals

best_test.insert(0, 'Fold', list(range(n_blocks)))

ave_best = best_test.mean()
ave_best.Fold = 'Average'

# best_test = best_test.append(ave_best, ignore_index=True)
best_test = pd.concat([best_test, ave_best.to_frame().T], ignore_index=True)

print(best_test)
# print(ave_best)
out.to_csv('output/x_validation2_rmse.csv')
best_test.to_csv('output/x_valid_final_rmse.csv')
# print(out)
