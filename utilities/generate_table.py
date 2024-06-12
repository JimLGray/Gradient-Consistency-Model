from context_class import Context
from data_io import fileIO, output
import regex
from context_class import Context, make_paramset
import numpy as np


def main():
    data_dir = 'data/'
    out_dir = 'output/scratch/'

    schedule_solves = [
        2,
        # 3, 4, 5,
        # 10, 20, 50, 100
    ]
    folders = [
        'stratified/',
        'training/',
        'additional/',
        # 'middlebury_2006/'
    ]

    folder_scenes = {
        'training/': [
            'boxes',
            'cotton',
            'dino',
            'sideboard'
        ],
        'additional/': [
            'antinous',
            'boardgames',
            'dishes',
            'greek',
            'kitchen',
            'medieval2',
            'museum',
            'pens',
            'pillows',
            'platonic',
            'rosemary',
            'table',
            'tomb',
            'tower',
            'town',
            'vinyl'
        ],
        'middlebury_2006/': [
            'Aloe',
            'Baby1',
            'Baby2',
            'Baby3',
            'Bowling1',
            'Bowling2',
            'Cloth1',
            'Cloth2',
            'Cloth3',
            'Cloth4',
            'Flowerpots',
            'Lampshade1',
            'Lampshade2',
            'Midd1',
            'Midd2',
            'Monopoly',
            'Plastic',
            'Rocks1',
            'Rocks2',
            'Wood1',
            'Wood2',
        ],
        'stratified/': [
            'backgammon',
            'dots',
            'pyramids',
            'stripes',
        ]
    }
    methods = [
        # 'old',
        # 'inter_irls',
        'gc_model',
        # 'prog_inc',
        # 'naive',
        # 'hybrid',
        # 'gc_no_cse',
        # 'gc_no_ge'
    ]
    alph_list = [
        # 0.0005,
        # 0.001,
        # 0.002,
        # 0.005,
        # 0.01,
        # 0.02,
        # 0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
    ]
    method_alphas = {
        'old': alph_list,
        'inter_irls': alph_list,
        'gc_model': alph_list,
        'hybrid': alph_list,
        'gc_no_cse': alph_list,
        'gc_no_ge': alph_list,
        'prog_inc': alph_list,
        'naive': alph_list,
    }

    for N in schedule_solves:
        for folder in folders:
            scenes = folder_scenes[folder]
            if folder == 'training/':
                table_name = 'table_training.csv'
            elif folder == 'additional/':
                table_name = 'table_additional.csv'
            elif folder == 'stratified/':
                table_name = 'table_stratified.csv'
            else:
                table_name = 'table.csv'
            ctx_list = gen_ctxes(
                scenes, methods, method_alphas, folder, data_dir, out_dir, N
            )
            solve_str = str(N) + '_solves/'
            table_str = out_dir + solve_str + folder + table_name
            output.res_table(ctx_list, table_str)



def gen_ctxes(
        scene_list: list,
        method_list: list,
        meth_alphas: dict,
        folder: str,
        data_dir: str,
        out_dir: str,
        n_solves_schedule: int
):
    ctx_list = []
    solve_str = str(n_solves_schedule) + '_solves/'
    for scene in scene_list:
        input_path = data_dir + folder + scene + '/'
        for method in method_list:
            alphas = meth_alphas[method]
            for alpha in alphas:
                output_path = \
                    out_dir + solve_str + folder + scene + '/' + \
                    method + '/' + 'alpha=' + str(alpha) + '/'
                parameters = make_paramset(
                    input_path, output_path, method, alpha, n_solves_schedule
                )
                con = gen_ctx(parameters)
                ctx_list.append(con)
    return ctx_list


def gen_ctx(param_set):
    con = Context()
    for param in param_set:
        if param == 'gt_name':
            gt_name = param_set[param]
            if 'middlebury_2006' in gt_name:
                gt = fileIO.open_middlebury2006_gt(gt_name)
            else:
                gt, _ = fileIO.readPFM(gt_name)
            con.gt_disparity = gt
        elif param == 'baselines':
            con.baselines = np.array(param_set[param])
        else:
            con.__setattr__(param, param_set[param])
    return con


if __name__ == "__main__":
    main()
