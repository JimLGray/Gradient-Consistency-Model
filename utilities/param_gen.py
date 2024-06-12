import json
import regex
from data_io import fileIO
from context_class import make_paramset


def main():
    data_dir = '/srv/scratch/z3459143/data/'
    folders = [
        # 'training/', 
        # 'additional/', 
        'stratified/',
        # 'middlebury_2006/'
    ]
    out_dir = '/srv/scratch/z3459143/output/'
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
        'old',
        # 'naive',
        # 'inter_irls',
        'gc_model',
        # 'hybrid',
        'gc_no_cse',
        'gc_no_ge',
        # 'prog_inc',
    ]
    alph_list =[
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
        'naive': alph_list,
        'gc_no_cse': alph_list,
        'gc_no_ge': alph_list,
        'prog_inc': alph_list,
    }

    schedule_solves_list = [
        2,
        # 3,
        # 4,
        # 5,
        # 10,
        # 20,
        # 50,
        # 100,
    ]

    idx = 0
    for N in schedule_solves_list:
        solve_title = str(N) + '_solves/'
        for folder in folders:
            scenes = folder_scenes[folder]
            for scene in scenes:
                input_path = data_dir + folder + scene + '/'
                for method in methods:
                    alphas = method_alphas[method]
                    for alpha in alphas:
                        output_path = \
                            out_dir + solve_title + folder + scene + '/' + \
                            method + '/' + 'alpha=' + str(alpha) + '/'
                        print(output_path)
                        con = make_paramset(input_path, output_path, method,
                                            alpha, N)
                        con_name = 'params/context_' + str(idx) + '.json'
                        with open(con_name, 'w') as con_file:
                            json.dump(con, con_file, indent=4)
                        idx += 1


if __name__ == "__main__":
    main()
