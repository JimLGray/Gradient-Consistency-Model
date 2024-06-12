"""
Opens a set of parameters and then runs the algorithm using the set of
parameters. This should be somewhat paired with param_gen.py as there are lots
of parameters in each parameter file.
"""

import json
import numpy as np
from context_class import Context
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

    con = Context()
    for param in parameters:
        if param == 'gt_name':
            gt_name = parameters[param]
            if 'middlebury_2006' in gt_name:
                gt = fileIO.open_middlebury2006_gt(gt_name)
            else:
                gt, _ = fileIO.readPFM(gt_name)
            con.gt_disparity = gt
        elif param == 'baselines':
            con.baselines = np.array(parameters[param])
        else:
            con.__setattr__(param, parameters[param])
    Path(con.res_dir).mkdir(parents=True, exist_ok=True)

    con_dict = con.to_dict()
    print(con_dict)

    mv.mv_disparity(con)


if __name__ == "__main__":
    main()
