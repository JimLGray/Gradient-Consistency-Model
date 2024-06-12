import adj_mat as am
import numpy as np
import pytest


def test_adj_mat_line():
    baselines = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    # answer is off-diagonal matrix downwards, with 2s on the off-diagonal
    ans_diag = np.full(3, 2)
    ans = np.diag(ans_diag, -1)
    res = am.adj_mat(baselines)
    assert np.allclose(ans, res)


def test_adj_mat_opp_sectors():
    baselines = np.array([[1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [-3, 0],
                          [4, 0], [-4, 0]])
    ans_diag = np.full(6, 2)
    ans = np.diag(ans_diag, -2)
    res = am.adj_mat(baselines)
    assert np.allclose(ans, res)


def test_adj_mat_adj_sectors():
    baselines = np.array([[1, 0], [-1, 0], [1, 1], [-2, 0], [2, 2]])
    ans_diag = np.array([1, 2, 2])
    ans = np.diag(ans_diag, -2)
    res = am.adj_mat(baselines)
    assert np.allclose(ans, res)


def test_chain_line_simple_3rd():
    baselines = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    sectors = np.ones(4)
    idx = 2
    ans = np.full_like(sectors, 0)
    ans[1] = 2
    res = am.chain(baselines, sectors, idx)
    assert np.allclose(ans, res)


def test_chain_one_sector_4th():
    baselines = np.array([[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2],
                          [4, 0]])
    sectors = np.ones(7)
    idx = 3
    ans = np.full_like(sectors, 0)
    ans[1:3] = 2
    res = am.chain(baselines, sectors, idx)
    assert np.allclose(ans, res)


def test_chain_opposite_sectors_4th():
    baselines = np.array([[1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [-3, 0],
                          [4, 0], [-4, 0]])
    sectors = np.array([1, 5, 1, 5, 1, 5, 1, 5])
    idx = 5
    ans = np.array([0, 0, 0, 2, 0, 0, 0, 0])
    res = am.chain(baselines, sectors, idx)
    assert np.allclose(ans, res)


def test_chain_adj_sectors_3rd():
    baselines = np.array([[1, 0], [-1, 0], [1, 1], [-2, 0]])
    sectors = np.array([1, 5, 2, 5])
    idx = 2
    ans = np.array([[1, 0, 0, 0]])
    res = am.chain(baselines, sectors, idx)
    assert np.allclose(ans, res)


def test_chain_adj_sectors_5th():
    baselines = np.array([[1, 0], [-1, 0], [1, 1], [-2, 0], [2, 2]])
    sectors = np.array([1, 5, 2, 5, 2])
    idx = 4
    ans = np.array([[0, 0, 2, 0, 0]])
    res = am.chain(baselines, sectors, idx)
    assert np.allclose(ans, res)


def test_sector_2_1_arr():
    baseline = np.array([1, 0])
    assert am.sector(baseline) == 0


def test_sector_1_1_tuple():
    baseline = (1, 1)
    assert am.sector(baseline) == 1


def test_sector_neg1_3_arr():
    baseline = np.array([-1, 3])
    assert am.sector(baseline) == 2


def test_sector_neg3_1_tuple():
    baseline = (-3, 1)
    assert am.sector(baseline) == 3


def test_sector_neg3_0_tuple():
    baseline = (-3, 0)
    assert am.sector(baseline) == 4


def test_sector_neg2_neg2_tuple():
    baseline = (-2, -2)
    assert am.sector(baseline) == 5


def test_sector_0_neg1_tuple():
    baseline = (0, -1)
    assert am.sector(baseline) == 6


def test_sector_2_neg2_list():
    baseline = [2, -2]
    assert am.sector(baseline) == 7


def test_sector_0_0_tuple():
    baseline = (0, 0)
    with pytest.raises(ValueError):
        am.sector(baseline)


def test_nearest_neighbour_line():
    baselines = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    idx = 2
    assert am.nearest_neighbour(baselines, idx) == 1


def test_nearest_neighbour_diagonal():
    baselines = np.array([[1, 0], [2, 1], [2, 2]])
    idx = 1
    assert am.nearest_neighbour(baselines, idx) == 2


def test_nearest_neighbour_big_dist():
    baselines = np.array([[0, 0], [4, 0], [3, 1], [2, 2]])
    baselines += 1
    idx = 0
    assert am.nearest_neighbour(baselines, idx) == 3


def test_neighbourhood_line():
    baselines = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    idx = 2
    assert am.neighbourhood(baselines, idx) == {1, 2, 3}


def test_neighbourhood_diagonal():
    baselines = np.array([[1, 0], [2, 1], [2, 2]])
    idx = 1
    assert am.neighbourhood(baselines, idx) == {0, 1, 2}


def test_neighbourhood_big_dist():
    baselines = np.array([[0, 0], [4, 0], [3, 1], [2, 2], [2, 0]])
    baselines += 1
    assert am.neighbourhood(baselines, 0) == {0, 3, 4}


def test_get_sector_all_sector_0():
    baselines = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    res = am.get_sector(baselines, 0)
    assert res == [0, 1, 2, 3]


def test_get_sector_half_2():
    baselines = np.array([[0, 1], [2, 0], [0, 3], [4, 0]])
    res = am.get_sector(baselines, 2)
    assert res == [0, 2]


def test_get_sector_all_1():
    baselines = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    res = am.get_sector(baselines, 1)
    assert res == [0, 1, 2, 3]


def test_get_sector_ref_first():
    baselines = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    res = am.get_sector(baselines, 1)
    assert res == [1, 2, 3, 4]


def test_get_sector_ref_first_wrong_sector():
    baselines = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    res = am.get_sector(baselines, 4)
    assert res == []


def test_invalid_sector():
    baselines = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    with pytest.raises(ValueError):
        am.get_sector(baselines, 9)
