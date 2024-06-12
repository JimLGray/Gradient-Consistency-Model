import json
import numpy as np
from context_class import Context, State
from data_io import fileIO, output
from pathlib import Path
import multiview as mv
import sys


def main():
    """

    Returns
    -------

    """
    param_name = sys.argv[1]
    with open(param_name, 'r') as param_file:
        parameters = json.load(param_file)

    first_con = Context()
    for param in parameters:
        if param == 'gt_name':
            gt_name = parameters[param]
            if 'middlebury_2006' in gt_name:
                gt = fileIO.open_middlebury2006_gt(gt_name)
            else:
                gt, _ = fileIO.readPFM(gt_name)
            first_con.gt_disparity = gt
        elif param == 'baselines':
            first_con.baselines = np.array(parameters[param])
        else:
            first_con.__setattr__(param, parameters[param])
    Path(first_con.res_dir).mkdir(parents=True, exist_ok=True)

    con_dict = first_con.to_dict()
    print(con_dict)

    w, state = mv.mv_disparity(first_con)

    second_con = first_con.copy()
    second_con.n_levels = 1
    second_con.multi_res_levels = 3
    second_con.total_max_warps = 1
    second_con.grad_err = True
    second_con.vmax = 2 ** 2
    second_con.vmin = 2 ** 2
    # state.total_ctf_levels += 1
    # state.total_warp_stages += 1
    state.ctf_level = 0

    w, state = mv.mv_disparity(second_con, state, w)


if __name__ == "__main__":
    main()
