import pytest
from regex import *
import numpy as np


def test_regex_slats_valid():
    filenames = ['A(0,1).kek', 'B(3,2).blah']
    coordinates, valid_names = regex_slats(filenames)
    ans = [(0, 1), (3, 2)]
    assert coordinates == ans
    assert valid_names == filenames


def test_regex_slats_invalid():
    filenames = ['ABC*&.png, 123[&.png']
    with pytest.raises(ValueError):
        regex_slats(filenames)


def test_regex_slats_partial():
    filenames = ['A(0,1).kek', 'B(3,2).blah', 'ABC.png, 123.png']
    coordinates, valid_names = regex_slats(filenames)
    ans = [(0, 1), (3, 2)]
    assert coordinates == ans
    assert valid_names == ['A(0,1).kek', 'B(3,2).blah']


def test_regex_gradients_valid():
    filenames = ['memes01.ry', 'doot21.help', 'test00.png']
    coordinates, valid_names = regex_gradients(filenames)
    ans = [(1, 0), (21, 0), (0, 0)]
    assert coordinates == ans
    assert valid_names == filenames


def test_regex_gradients_invalid():
    filenames = ['memesAB.ry', 'doo[t.,.help']
    with pytest.raises(ValueError):
        regex_gradients(filenames)


def test_regex_gradients_partial():
    filenames = ['memesAB.ry', 'memes01.ry', 'doot21.help', 'doo[t.,.help']
    coordinates, valid_names = regex_gradients(filenames)
    ans = [(1, 0), (21, 0)]
    assert coordinates == ans
    assert valid_names == ['memes01.ry', 'doot21.help']


# def test_regex_lightfield_valid():
#     filenames = ['test040.png', '^$jff004.png', 'img80.png', 'img08.png', 'help00.png']
#     coordinates, valid_names = regex_light_field(filenames)
#     ans = [(0, 0), (0, 4), (4, -4), (4, 4), (-4, 4)]
#     assert coordinates == ans
#     assert valid_names == filenames


def test_regex_lightfield_invalid():
    filenames = ['memesAB.ry', 'doo[t.,.help']
    with pytest.raises(ValueError):
        regex_light_field(filenames)


# def test_regex_lightfield_partial():
#     filenames = ['memesAB.ry', 'test040.ry', '$jff004.help', 'doo[t.,.help']
#     coordinates, valid_names = regex_light_field(filenames)
#     ans = [(0, 0), (0, 4)]
#     assert coordinates == ans
#     assert valid_names == ['test040.ry', '$jff004.help']

def test_sort_baselines_names_positive():
    baselines = np.array([[0, 0], [0, 5], [2, 2], [3, 3], [1, 1], [1, 0]])
    names = [
        'ref',
        'tgt05',
        'tgt22',
        'tgt33',
        'tgt11',
        'tgt10',
    ]
    ans_blines = np.array([[0, 0], [1, 0], [1, 1], [2, 2], [3, 3], [0, 5]])
    ans_names = [
        'ref',
        'tgt10',
        'tgt11',
        'tgt22',
        'tgt33',
        'tgt05'
    ]
    sorted_b, sorted_n = sort_baselines_names(baselines, names)
    assert np.allclose(sorted_b, ans_blines)
    assert (sorted_b == ans_blines).all()


def test_sort_baselines_names_negative():
    baselines = -np.array([[0, 0], [0, 5], [2, 2], [3, 3], [1, 1], [1, 0]])
    names = [
        'ref',
        'tgt(0, -5)',
        'tgt(-2, -2)',
        'tgt(-3, -3)',
        'tgt(-1, -1)',
        'tgt(-1, 0)'
    ]
    ans_blines = -np.array([[0, 0], [1, 0], [1, 1], [2, 2], [3, 3], [0, 5]])
    ans_names = [
        'ref',
        'tgt(-1, 0)',
        'tgt(-1, -1)',
        'tgt(-2, -2)',
        'tgt(-3, -3)',
        'tgt(0, -5)'
    ]
    sorted_b, sorted_n = sort_baselines_names(baselines, names)
    assert np.allclose(sorted_b, ans_blines)
    assert (sorted_b == ans_blines).all()


def test_sort_baselines_names_mixed():
    baselines = np.array([[0, 0], [0, -5], [2, 2], [3, -3], [1, 1], [-1, 0]])
    names = [
        'ref',
        'tgt(0, -5)',
        'tgt(2, 2)',
        'tgt(-3, -3)',
        'tgt(1, 1)',
        'tgt(-1, 0)'
    ]
    ans_blines = np.array([[0, 0], [-1, 0], [1, 1], [2, 2], [3, -3], [0, -5]])
    ans_names = [
        'ref',
        'tgt(-1, 0)',
        'tgt(1, 1)',
        'tgt(2, 2)',
        'tgt(3, -3)',
        'tgt(0, -5)'
    ]
    sorted_b, sorted_n = sort_baselines_names(baselines, names)
    assert np.allclose(sorted_b, ans_blines)
    assert (sorted_b == ans_blines).all()
