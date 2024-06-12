# Module regex.py

import re
import ast
import math
import numpy as np


def regex_slats(filenames):
    """
    Takes a list of strings and then returns a list of baselines associated with
    the strings. The expected pattern is blah(0, 1).png, and the bit inside the
    brackets becomes the coordinate or the baseline
    :param filenames: must an iterable of strings
        the filenames to find the baseline from
    :return:
        a list of tuples which contain the coordinates

    """

    # basically looks for (-a, b) sort of expresssions:
    # https://regex101.com/r/xEiGXo/1/ for more details
    regex = r"[(][-]?\d*[,][-]?\d*[)]"
    prog = re.compile(regex)

    baselines_list = []
    valid_names = []

    for name in filenames:
        res = re.search(prog, name)
        if res:
            coordinate_str = res[0]
            baselines_list.append(ast.literal_eval(coordinate_str))
            valid_names.append(name)

    if len(baselines_list) == 0:
        raise ValueError('No valid filenames given')

    return baselines_list, valid_names


def regex_gradients(filenames):
    """
    Takes a list of strings and then returns a list of baselines and associated
    with the strings. The expected pattern is blah01.png where the digits are
    baseline in the x direction.

    :param filenames: must an iterable of strings
        the filenames to find the baseline from
    :return:
        a list of tuples which contain the coordinates
    """
    search_regex = r"\d+"
    prog = re.compile(search_regex)

    baselines_list = []
    valid_names = []
    for name in filenames:
        res = re.search(prog, name)
        if res:
            coordinate_str = res[0]
            coordinate_str = re.sub(r'\b0+', '', coordinate_str)
            if coordinate_str == '':
                x_coordinate = 0
            else:
                x_coordinate = ast.literal_eval(coordinate_str)
            baselines_list.append((x_coordinate, 0))
            valid_names.append(name)

    if len(baselines_list) == 0:
        raise ValueError('No valid filenames given')

    return baselines_list, valid_names


def regex_light_field(filenames):
    """
    Takes a list of strings and then returns a list of baselines and associated
    with the strings. The expected pattern is blah01.png where the digits
    provide a number which determines the baseline. it's a 9x9 grid with 00
    starting in the top left corner.

    Parameters
    ----------
    filenames: list of strings
        List of filename to open

    Returns
    -------
    baselines_list:
        a list of baseline tuples
    valid_names:
        a list of strings of the corresponding names
    """
    search_regex = r"\/input_Cam\d+"
    prog = re.compile(search_regex)

    baselines_list = []
    valid_names = []
    for name in filenames:
        res = re.search(prog, name)
        if res:
            img_num_str = res[0][11:]
            img_num_str = re.sub(r'\b0+', '', img_num_str)
            if img_num_str == '':
                img_num = 0
            else:
                img_num = ast.literal_eval(img_num_str)
            x_coordinate = img_num % 9 - 4
            y_coordinate = 4 - math.floor(img_num / 9)
            baselines_list.append((y_coordinate, x_coordinate))
            valid_names.append(name)

    if len(baselines_list) == 0:
        raise ValueError('No valid filenames given')

    return baselines_list, valid_names


def regex_2013_lf_data(filenames: list):
    """
    Takes a list of strings and then returns a list of baselines and associated
    with the strings. The expected pattern is (x, y).png where the x and y
    values determine the baseline.

    Parameters
    ----------
    filenames: list of strings
        List of filename to open

    Returns
    -------
    baselines_list:
        a list of baseline tuples, in (y,x) order
    valid_names:
        a list of strings of the corresponding names
    """
    baselines_list = []
    valid_names = []
    idx = 0

    search_regex = '(\(-?\d+)|(-?\d+\))'

    for name in filenames:
        idx += 1
        res = re.findall(search_regex, name)
        if len(res) == 2:
            x_str = res[0][0]
            x_val = ast.literal_eval(x_str[1:])
            y_str = res[1][1]
            y_val = ast.literal_eval(y_str[0:-1])
            baselines_list.append((y_val, x_val))
            valid_names.append(name)

    if len(baselines_list) == 0:
        raise ValueError('No valid filenames given')

    return baselines_list, valid_names



def sort_baselines_names(baselines: np.ndarray | list, filenames: list):
    """
    Sorts the baselines and names such that small baselines are first and so are
    their corresponding names

    Parameters
    ----------
    baselines
    filenames

    Returns
    -------
    sorted_baselines, sorted_names
    """
    pair_list = list(zip(baselines, filenames))

    def get_bline_mag(item):
        bline = item[0]
        b_mag = bline[0] ** 2 + bline[1] ** 2
        return b_mag

    pair_list.sort(key=get_bline_mag)
    sorted_baselines, sorted_names = list(zip(*pair_list))
    return sorted_baselines, sorted_names
