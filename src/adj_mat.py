# Module adj_mat.py
"""
All of these functions are built on the assumption that the images are in a grid
like pattern. Maybe with some grid entries missing.
"""
from __future__ import annotations
import numpy as np


def adj_mat(baselines: np.ndarray | list):
    """
    Creates an adjacency matrix for baselines. A baseline is adjacent to another
    baseline if the other baseline is closer to the reference view than the
    baseline in question and it must be in the same sector and neighbourhood.
    These are added in a "worst of" way

    If there are no other images in the same sector and neighbourhood, but the
    image is not adjacent to the reference view then we add views from adjacent
    sectors to the chain in a "best of" way.

    Best of is 1, worst of is 2

    Parameters
    ----------
    baselines: np.ndarray | list

    Returns
    -------
    np.ndarray
        A 2D adjacency matrix.
    """

    n_pairs = len(baselines)

    sectors = list(map(sector, baselines))

    mat = np.empty((n_pairs, n_pairs))
    for p in range(n_pairs):
        mat[p, :] = chain(baselines, sectors, p)

    return mat


def chain(baselines: np.ndarray | list | tuple, sectors: np.ndarray | list,
          idx: int):
    """
    Determines which baselines are in the chain. For another baseline to be in
    the same chain, it must be closer than the baseline in question, and it must
    be in the same sector and neighbourhood. These are added in a "worst of"
    way.

    If there are no other images in the same sector and neighbourhood, but the
    image is not adjacent to the reference view then we add views from adjacent
    sectors to the chain in a "best of" way.

    Best of is 1, worst of is 2

    Parameters
    ----------
    baselines: np.ndarray | list | tuple
        baselines of the views. must be sorted in ascending order in terms of
        magnitude.
    sectors: np.ndarray | list
        sector of view in question
    idx: int
        the index in the baselines and the sector of the view in question.


    Returns
    -------
    np.ndarray
        An array of the same length as baseline, indicating which
        baselines are in the chain and in which form.

    """

    n_pairs = len(baselines)

    if n_pairs != len(sectors):
        raise ValueError('Baselines and sectors must be equal in length')

    if 0 > idx >= n_pairs:
        raise ValueError('Invalid index')

    chain_arr = np.zeros(n_pairs)

    if idx == 0:
        return chain_arr

    found = False
    sec_val = sectors[idx]
    sec_big = (sec_val + 1) % 8
    sec_sml = (sec_val - 1) % 8

    neighbours = neighbourhood(baselines, idx)

    for p in range(0, idx):
        if sectors[p] == sec_val and p in neighbours:
            chain_arr[p] = 2
            found = True

    if not found:
        for p in range(0, idx):
            sec = sectors[p]
            if sec == sec_sml or sec == sec_big and p in neighbours:
                chain_arr[p] = 1

    return chain_arr


def get_sector(baselines: np.ndarray | tuple | list, sec: int):
    """
    Gets a list of baselines from the sector

    Sector list is:
    V_0 = [0, \pi/4) \\
    V_1 = [\pi/4, \pi/2) \\
    V_2 = [\pi/2, 3\pi/4) \\
    V_3 = [3\pi/2, \pi) \\
    V_4 = [\pi, -3\pi/4) \\
    V_5 = [-3\pi/4, -\pi/2) \\
    V_6 = [-\pi/2, -\pi/4) \\
    V_7 = [-\pi/4, 0)

    Parameters
    ----------
    baselines: np.ndarray | list | tuple
        Must have length at least two.
    sec: int
        Value must be between 0 and 7 inclusive.

    Returns
    -------
    list
        a list of the baselines from that sector.
    """
    if 0 > sec or sec > 8:
        raise ValueError('Invalid Sector Choice')
    if type(sec) != int:
        if type(sec) == float:
            if sec.is_integer():
                s = int(sec)
        else:
            raise ValueError('sec must be a integer')
    else:
        s = sec

    idx = 0
    ret_list = []
    for b in baselines:
        try:
            sec_val = sector(b)
            if sec_val == s:
                ret_list.append(idx)
        except ValueError:
            pass
        idx += 1
    return ret_list


def sector(baseline: np.ndarray | tuple | list):
    """
    Sector list is:
    V_0 = [0, \pi/4) \\
    V_1 = [\pi/4, \pi/2) \\
    V_2 = [\pi/2, 3\pi/4) \\
    V_3 = [3\pi/2, \pi) \\
    V_4 = [\pi, -3\pi/4) \\
    V_5 = [-3\pi/4, -\pi/2) \\
    V_6 = [-\pi/2, -\pi/4) \\
    V_7 = [-\pi/4, 0)

    if 0,0, return sector raise error, because not valid.

    Parameters
    ----------
    baseline: np.ndarray | list | tuple
        Must have length at least two.

    Returns
    -------
    int
        The sector number.

    """
    b_0 = baseline[0]
    b_1 = baseline[1]

    if b_0 == 0 and b_1 == 0:
        raise ValueError('Baseline of (0,0) cannot be placed in a sector')

    ang = np.arctan2(b_1, b_0)
    if ang < 0:
        ang = ang + 2 * np.pi
    tmp = ang / (np.pi / 4)
    sec = int(np.floor(tmp))
    return sec


def neighbourhood(vectors: np.ndarray | list, idx: int):
    """
    Finds the indexes of the vectors which are in the local neighbourhood
    to the vector in question as specified by idx.

    A neighbourhood is defined as having the same max norm as the nearest view.

    Parameters
    ----------
    vectors: np.ndarray | list
        List or array of baselines from which to choose from
    idx: int
        The index of the baseline in question.

    Returns
    -------
    n_idxes: set
        A set of the indexes which are in the neighbourhood. Of varying size.
        Includes the baseline in question

    """
    nn_idx = nearest_neighbour(vectors, idx)

    n_pairs = len(vectors)
    if 0 > idx >= n_pairs:
        raise ValueError('Invalid index')

    v1 = vectors[idx]
    nn = vectors[nn_idx]
    thresh = np.maximum(np.abs(v1[0] - nn[0]), np.abs(v1[1] - nn[1]))

    n_idxes = set()

    for p in range(n_pairs):
        if p == idx or p == nn_idx:
            n_idxes.add(p)
        else:
            v2 = vectors[p]
            max_norm = np.maximum(np.abs(v1[0] - v2[0]), np.abs(v1[1] - v2[1]))
            if max_norm <= thresh:
                n_idxes.add(p)
    return n_idxes


def nearest_neighbour(vectors: np.ndarray | list, idx: int):
    """
    Finds the nearest neighbour from a list of baselines. The nearest is taken
    in the euclidean sense.

    If there is a tie, we take the first one that appears.

    Parameters
    ----------
    vectors: np.ndarray | list
        List or array of baselines from which to choose from
    idx: int
        The index of the baseline in question.

    Returns
    -------
    nn_idx: idx
        index of the nearest neighbour
    """

    n_pairs = len(vectors)
    if 0 > idx >= n_pairs:
        raise ValueError('Invalid index')

    v1 = vectors[idx]
    min_dist = np.inf

    nn_idx = -1
    for p in range(n_pairs):
        if p == idx:
            continue
        else:
            v2 = vectors[p]
            dist = (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2
            if min_dist > dist:
                min_dist = dist
                nn_idx = p

    return nn_idx
