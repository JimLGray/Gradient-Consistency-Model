# Module cost.py

import numpy as np
import graph


def point_wise_cost(W_pq: np.ndarray,
                    data_irls_weights: np.ndarray,
                    g_pq: np.ndarray,
                    delta_Ipq: np.ndarray, w: np.ndarray, alpha):
    """
    Calculates the cost using the objective function, pointwise.

    Parameters
    ----------
    W_pq: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains W_pq(s)
    data_irls_weights: np.ndarray

    g_pq: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains g_pq(s)
    delta_Ipq: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains delta I_pq(s)
    w: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains w(s), the
        disparity
    alpha: float
        regularisation constant.


    Returns
    -------

    """
    e_d = np.sum(W_pq * data_irls_weights * (g_pq * w + delta_Ipq) ** 2,
                 axis=(0, 1))
    w_x, w_y = np.gradient(w)
    e_r = np.sqrt(w_x ** 2 + w_y ** 2) * np.sum(W_pq, axis=(0, 1)) * alpha
    return e_d, e_r


def data_cost(weights: np.ndarray, dot_grads: np.ndarray,
                     diff: np.ndarray, w: np.ndarray):
    """
    Calculates the data cost over every pixel for each scale and each view.
    Implements the equation

    sum_omega sum_pq W_pq(s) * (g_pq(s) + delta I_pq(s))

    Parameters
    ----------
    weights: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains W_pq(s)
    dot_grads: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains g_pq(s)
    diff: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains delta I_pq(s)
    w: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains w(s), the
        disparity

    Returns
    -------
    E_d: float
        The data cost.
    """
    e_d = weights * (dot_grads * w + diff) ** 2
    return np.sum(e_d)


def welsch(x: np.ndarray, sigma: float):
    return sigma ** 2 * (1 - np.exp(-x**2 / (2 * sigma ** 2)))


def reg_cost(w: np.ndarray):
    """
    Calculates the regularisation cost. Essentially
    sum
    Parameters
    ----------
    w: np.ndarray
        [n_targets, n_res, height, width] in dimensions. Contains w(s), the
        disparity

    Returns
    -------
    E_r: float
        The regularisation cost
    """
    w_x, w_y = np.gradient(w)
    e_r = np.sqrt(w_x ** 2 + w_y ** 2)
    return np.sum(e_r)


def cg_cost(sum_g_pq: np.ndarray, reg_graph: graph.FullWeightedGraph,
            b: np.ndarray, w: np.ndarray):
    """
    Calculates

    w^T A w / 2 - b^T w

    Parameters
    ----------
    sum_g_pq
    reg_graph
    b
    w

    Returns
    -------

    """
    Aw = sum_g_pq * w + reg_graph.graph_sum(w)
    wAw = np.sum(w * Aw) / 2
    return wAw - np.sum(b * w)
