# Module cgrad.pyx

import numpy as np
cimport numpy as np
cimport cython
import graph
from libc.math cimport fabs

DTYPE = np.float64
ctypedef  np.float64_t DTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cpdef np.ndarray[DTYPE_t, ndim=2] cg_disparity(
        np.ndarray[DTYPE_t, ndim=2] w,
        np.ndarray[DTYPE_t, ndim=2] dw,
        np.ndarray[DTYPE_t, ndim=2] sum_g_pq,
        np.ndarray[DTYPE_t, ndim=2] b,
        object reg_graph,
        int iterations,
        double tol=1e-4
):
    """
    Uses the conjugate gradient method to calculate the disparity. We use 
    notation as per the book Conjugate Gradient Without the Agnozing Pain.
    
    Parameters
    ----------
    dw: np.ndarray
        Dimensions: [height, width]
    sum_g_pq: np.ndarray
        Weighted sum of all the g_pqs for all of p and q 
        Dimensions [height, width]
    b: np.ndarray
        Weighted sum of all the delta_Ipqs for all of p and q.
        Dimensions [height, width]
    reg_graph: graph.FullWeightedGraph
        This has to be after weighting and applying the kernel.
    iterations: int
        number of iterations to use
    tol: float (optional)
        default is 1e-4
    
    Returns
    -------
    disparity: np.ndarray
        estimated disparity.
    """
    x = np.copy(dw)
    Ax = sum_g_pq * x + reg_graph.graph_sum(x + w)
    r = b - Ax
    d = np.copy(r)

    cdef int height = sum_g_pq.shape[0]
    cdef int width = sum_g_pq.shape[1]

    if np.amax(np.abs(r)) < tol:
        return x

    cdef double rTr = np.sum(r ** 2)
    cdef double alpha, beta, rTr_new

    cdef int n
    cg_disparity_loop(w, x, r, d, sum_g_pq, reg_graph, rTr, tol, iterations,
                      height, width)

    return x

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # Use C-division where we don't check divide by zero error
cdef void cg_disparity_loop(
        double[:, :] w,
        double[:, :] x,
        double[:, :] r,
        double[:, :] d,
        double[:, :] g_pq,
        object reg_graph,
        double rTr,
        double tol,
        int iterations,
        int height,
        int width
):
    """
    Does the loop inside cg_disparity ideally quite fast
    
    Parameters
    ----------
    x
    r
    d
    rTr
    tol
    iterations

    Returns
    -------

    """

    cdef int n, i, j, s_1, s_2
    cdef double g_val
    cdef double alpha, beta, rTr_new, dAd,

    Ad_arr = np.empty((height, width))
    cdef double[:, :] Ad = Ad_arr
    rg_sum_arr = np.empty((height, width))
    cdef double[:, :] rg_sum = rg_sum_arr
    abs_r_arr = np.abs(r)
    cdef double[:, :] abs_r = abs_r_arr
    cdef int small_enough

    reg_g_arr = reg_graph.graph
    cdef double[:, :, :, :] g = reg_g_arr
    cdef double g_val_left, g_val_right, g_val_top, g_val_bot, g_val_cnr

    for n in range(iterations):
        reg_graph.g_sum_inplace(d, rg_sum)
        dAd = 0.
        for i in range(height):
            for j in range(width):
                Ad[i, j] = g_pq[i, j] * d[i, j] + rg_sum[i, j]

        for i in range(height):
            for j in range(width):
                dAd += d[i, j] * Ad[i, j]
        alpha = rTr / dAd

        rTr_new = 0.
        for i in range(height):
            for j in range(width):
                x[i, j] += alpha * d[i, j]
                r[i, j] -= alpha * Ad[i, j]
                rTr_new += r[i, j] ** 2
                abs_r[i, j] = fabs(r[i, j])

        beta = rTr_new / rTr

        for i in range(height):
            for j in range(width):
                d[i, j] = r[i, j] + beta * d[i, j]

        rTr = rTr_new

        small_enough = 1
        for i in range(height):
            for j in range(width):
                if abs_r[i, j] > tol:
                    small_enough = 0
                    break
            if small_enough == 0:
                break
                    # Early exit condition...
        if small_enough == 1:
            return
