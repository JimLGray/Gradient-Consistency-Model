# Module graph.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fmax, fabs, sqrt
from scipy import ndimage

DTYPE = np.float64
ctypedef  np.float64_t DTYPE_t

class FullWeightedGraph:
    """
    A weighted graph based on a numpy array. We store all of the weights between
    nodes in a 3x3 grid at each pixel.
    It looks like
    00 01 02
    10 11 12
    20 21 22
    Where the middle value 11 is blank

    Also, will be done using a zero order boundary extension process.
    """
    ___slots__ = ('height', 'width', 'graph', 'sigma_r', 'a', 'b',)

    def __init__(self, int height, int width, double sig_r=1,
                 double a=10, double b=1):
        self.height = height
        self.width = width
        self.graph = np.ones((height, width, 3, 3), dtype=np.double)
        self.sigma_r = sig_r
        self.a = a
        self.b = b

    def set_weights(self, np.ndarray[DTYPE_t, ndim=2] img, double sigma,
                    cost_func='welsch'):
        """

        Parameters
        ----------
        img
        sigma

        Returns
        -------

        """
        img_height = img.shape[0]
        img_width = img.shape[1]

        if (img_height, img_width) != (self.height, self.width):
            print('Image dimensions are not correct')
            return
        pad_width = 1
        pad_img = np.pad(img, pad_width, mode='edge')
        if cost_func == 'welsch' or cost_func == 'Welsch':
            # self.sigma_r = np.full((self.height, self.width), sigma)
            _set_welsch_weights(self.graph, pad_img, self.sigma_r, img_height,
                                img_width, pad_width)
        elif cost_func == 'L1':
            _set_l1_weights(self.graph, pad_img, img_height, img_width,
                            pad_width)
        else:
            print(cost_func)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def set_weights_4_neighbours_l1(self, np.ndarray[DTYPE_t, ndim=2] w, int q,
                                    dis_boundary_mode='edge',
                                    res_boundary_mode='edge'):
        cdef int w_height = w.shape[0]
        cdef int w_width = w.shape[1]
        cdef int pad_width = 1
        if (w_height, w_width) != (self.height, self.width):
            print('Image dimensions are not correct')
            return
        pad_w = np.pad(w, pad_width, mode=dis_boundary_mode)
        g = np.copy(self.graph)
        res = np.empty_like(w)
        _agg_l1_weights_4_neighbours(pad_w, res, w_height, w_width, pad_width)

        pad_res = np.pad(res, pad_width, mode=res_boundary_mode)
        # careful here.
        cdef double[:, :] p_r = pad_res

        new_g = np.zeros_like(g)
        cdef double[:, :, :, :] n_g = new_g
        set_4_neighbours_graph_weights(pad_res, n_g, w_height, w_width,
                                       pad_width, q)

        self.graph = new_g

    def apply_w_kernel(self, double reg_const, four_neighbours=True):
        """
        applies the default kernel to all the weights.
        Middle value is now used.
        """
        cdef double Q, R

        if four_neighbours:
            Q = -reg_const
            R = 0.0
        else:
            Q = -reg_const ** 2 / 6
            R = -reg_const ** 2 / 12

        cdef np.ndarray[DTYPE_t, ndim=2] w_kernel = np.array([[R, Q, R],
                                                               [Q, 0, Q],
                                                               [R, Q, R]])

        self.graph *= w_kernel
        s = np.sum(self.graph, axis=3)
        s = np.sum(s, axis=2)
        self.graph[:, :, 1, 1] = -s

    def normalise(self, d_weights, four_neighbours=True):
        """
        Uses the weights for the data term to normalise the regularisation
        weights

        Parameters
        ----------
        d_weights
        four_neighbours

        Returns
        -------

        """

        if not four_neighbours:
            raise Exception('Only works with fourneighbours scheme for now.')

        cdef int n_targets = d_weights.shape[0]
        cdef int n_res = d_weights.shape[1]

        cdef int q

        # scaled_weights = np.empty_like(d_weights)
        # for q in range(n_res):
            # Seeing as we scale the image data from these scales
            # we need to compensate for that. We don't scale the weights.
            # scaled_weights[:, q] = d_weights[:, q] * 2 ** q

        s_weights = np.sum(d_weights, axis=(0, 1)) / (n_res * n_targets)
        arr = np.array([0.5, 0.5, 0.])
        g_r = ndimage.convolve1d(s_weights, arr)
        g_l = np.copy(s_weights)
        g_l[:, 1:] = g_r[:, :-1]
        g_d = ndimage.convolve1d(s_weights, arr, axis=0)
        g_u = np.copy(s_weights)
        g_u[1:] = g_d[:-1]
        self.graph[:, :, 0, 1] *= g_u
        self.graph[:, :, 2, 1] *= g_d
        self.graph[:, :, 1, 0] *= g_l
        self.graph[:, :, 1, 2] *= g_r


    def graph_sum(self, double[:, :] img):
        """
        :param img:
        :return:
        """
        cdef int img_height = img.shape[0]
        cdef int img_width = img.shape[1]
        if (img_height != self.height) or (img_width != self.width):
            print('Image dimensions are not correct')
            return

        cdef int pad_width = 1
        pad_img = np.pad(img, pad_width, mode='edge')
        cdef double[:, :, :, :] g = self.graph
        cdef double[:, :] pad_view = pad_img
        cdef np.ndarray[DTYPE_t, ndim=2] g_sum = weighted_sum(
            g, pad_view, img_height, img_width, pad_width
        )

        return g_sum

    def g_sum_inplace(self, double[:, :] img, double[:, :] res):
        cdef int img_height = img.shape[0]
        cdef int img_width = img.shape[1]
        if (img_height != self.height) or (img_width != self.width):
            print('Image dimensions are not correct')
            return

        cdef int pad_width = 1
        pad_img = np.pad(img, pad_width, mode='edge')
        cdef double[:, :, :, :] g = self.graph
        cdef double[:, :] pad_view = pad_img
        weighted_sum_inplace(g, pad_view, res, img_height, img_width, pad_width)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.ndarray[DTYPE_t, ndim=2] weighted_sum(double[:, :, :, :] g,
                                              double[:, :] pad_img, int height,
                                              int width, int pad_width):
    """
    For each pixel (r, c) in pad_img, we select a 3x3 pixel neighbourhood around
    that pixel. We take the (r, c) entry in g which is a 3x3 array. We call this
    g[r,c]. The g[r, c] is multiplied elementwise by the 3x3 neighbourhood of 
    pad_img[r, c].

    Parameters
    ----------
    g
        graph
    pad_img
        padded image
    height
        height of the graph
    width
        width of the graph
    pad_width
        width of boundary extension

    Returns
    -------

    """
    cdef np.ndarray[DTYPE_t, ndim=2] g_sum = np.empty((height, width),
                                                      dtype=np.double)
    cdef double[:, :] g_sum_view = g_sum
    cdef double s = 0
    cdef double[:, :] t
    cdef int r, c, i, j
    for r in range(pad_width, height + pad_width):
        for c in range(pad_width, width + pad_width):
            s = 0
            for i in range(0, 3):
                for j in range(0, 3):
                    s = s + pad_img[r - 1 + i, c - 1 + j] \
                        * g[r - pad_width, c - pad_width, i, j]
            g_sum_view[r - pad_width, c - pad_width] = s
    return g_sum


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void weighted_sum_inplace(
        double[:, :, :, :] g,
        double[:, :] pad_img,
        double[:, :] res,
        int height,
        int width,
        int pad_width
):
    """
    For each pixel (r, c) in pad_img, we select a 3x3 pixel neighbourhood around
    that pixel. We take the (r, c) entry in g which is a 3x3 array. We call this
    g[r,c]. The g[r, c] is multiplied elementwise by the 3x3 neighbourhood of 
    pad_img[r, c].

    Parameters
    ----------
    g
        graph
    pad_img
        padded image
    height
        height of the graph
    width
        width of the graph
    pad_width
        width of boundary extension

    Returns
    -------

    """
    cdef double s = 0
    cdef double[:, :] t
    cdef int r, c, i, j
    for r in range(pad_width, height + pad_width):
        for c in range(pad_width, width + pad_width):
            s = 0
            for i in range(0, 3):
                for j in range(0, 3):
                    s = s + pad_img[r - 1 + i, c - 1 + j] \
                        * g[r - pad_width, c - pad_width, i, j]
            res[r - pad_width, c - pad_width] = s


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void _set_l1_weights(np.ndarray[DTYPE_t, ndim=4] g,
                              np.ndarray[DTYPE_t, ndim=2] pad_img,
                              int height, int width, int pad_width):
    """
    The weights for each vertex on the graph are determined by the difference
    between the image value at adjacent nodes or pixels. . 

    For the x[m] pixel in the neighbourhood around pixel x[n], the weight is 
    determined by:
    $1/max(|x[m] - x[n]|, 0.0001)$

    Parameters
    ----------
    g : np.ndarray
        Fullweighted Internal Graph (height, width, 3, 3) dimensions
    pad_img : np.ndarray
        Image padded by pad_wdith
    pad_width : int
        integer which describes how much pad_img has been padded by
    Returns
    -------
    nothing

    g is changed inplace to increase speed.

    """
    cdef int r, c, i, j, r_original, c_original
    cdef double delta = 1e-8
    cdef double x
    cdef double mu
    for r in range(pad_width, height + pad_width):
        for c in range(pad_width, width + pad_width):
            r_original = r - pad_width
            c_original = c - pad_width
            for i in range(0, 3):
                for j in range(0, 3):
                    x = pad_img[r - 1 + i, c - 1 + j]
                    mu = pad_img[r, c]
                    # g[r_original, c_original, i, j] = 1 / fmax(delta, fabs(x - mu))
                    g[r_original, c_original, i, j] = 1 / sqrt((x-mu) ** 2 + delta)
            g[r_original, c_original, 1, 1] = 0


def debug_set_4_neighbours_graph_weights(pad_weights, g_res, height, width,
                                         pad_width, q):
    return set_4_neighbours_graph_weights(pad_weights, g_res, height, width,
                                          pad_width, q)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void set_4_neighbours_graph_weights(
        double[:, :] pad_weights,
        double[:, :, :, :] g_res,
        int height,
        int width,
        int pad_width,
        int q
):
    cdef int r, c, i, j, r_pad, c_pad
    cdef double tmp, A, B, C, D
    cdef double h_l = 2.0 ** q
    print('scale parameter:', h_l)
    cdef double scale = 1.0 / (2.0 * h_l ** 2)
    for r in range(height):
        for c in range(width):
            r_pad = r + pad_width
            c_pad = c + pad_width
            tmp = pad_weights[r_pad, c_pad]
            A = (tmp + pad_weights[r_pad - 1, c_pad]) * scale
            B = (tmp + pad_weights[r_pad, c_pad + 1]) * scale
            C = (tmp + pad_weights[r_pad + 1, c_pad]) * scale
            D = (tmp + pad_weights[r_pad, c_pad - 1]) * scale
            g_res[r, c, 0, 1] = A
            g_res[r, c, 1, 2] = B
            g_res[r, c, 2, 1] = C
            g_res[r, c, 1, 0] = D


def debug_agg_l1_weights_4_neighbours(pad_w, res, height, width, pad_width):
    return _agg_l1_weights_4_neighbours(pad_w, res, height, width, pad_width)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void _agg_l1_weights_4_neighbours(
        np.ndarray[DTYPE_t, ndim=2] pad_w,
        np.ndarray[DTYPE_t, ndim=2] res,
        int height, int width, int pad_width):
    """
    For the w[s] pixel, the weight is determined by:
    $1/sqrt(|w_x[s] ^2 + w_y[s] ^2 | + delta)$
    w_x[r, c] = w[r, c + 1] - w[r, c - 1]
    w_y[r, c] = w[r + 1, c] - w[r - 1, c]

    Parameters
    ----------
    pad_w : np.ndarray
        padded disparity.
    res : np.ndarray
        the result array.
    pad_width : int
        integer which describes how much pad_img has been padded by
    Returns
    -------
    nothing

    g is changed inplace to increase speed.

    """
    cdef int r, c, i, j, r_original, c_original
    cdef double delta = 1e-7
    cdef double w_x, w_y
    for r in range(pad_width, height + pad_width):
        for c in range(pad_width, width + pad_width):
            r_og = r - pad_width
            c_og = c - pad_width
            w_y = (pad_w[r + 1, c] - pad_w[r - 1, c])/2.0
            w_x = (pad_w[r, c + 1] - pad_w[r, c - 1])/2.0
            res[r_og, c_og] = 1 / (2.0 * sqrt(w_x ** 2 + w_y ** 2 + delta))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void _set_welsch_weights(np.ndarray[DTYPE_t, ndim=4] g,
                              np.ndarray[DTYPE_t, ndim=2] pad_img,
                              double sigma,
                              int height, int width, int pad_width):
    """
    The weights for each vertex on the graph are determined by the difference 
    between the image value at adjacent nodes  or pixels. The weights are
    determined using a gaussian. 

    For the x[m] pixel in the neighbourhood around pixel x[n], the weight is 
    determined by:
    $G_\sig(x[m] - x[n])$

    Parameters
    ----------
    g : np.ndarray
        Fullweighted Internal Graph (height, width, 3, 3) dimensions
    pad_img : np.ndarray
        Image padded by pad_wdith
    sigma : np.ndarray
        local sigma value the gaussians used.
    pad_width : int
        integer which describes how much pad_img has been padded by
    Returns
    -------
    nothing

    g is changed inplace to increase speed.

    """
    cdef int r, c, i, j, r_og, c_og
    cdef double x
    cdef double mu
    cdef double temp
    cdef double[:, :, :, :] g_view = g

    for r in range(pad_width, height + pad_width):
        for c in range(pad_width, width + pad_width):
            r_og = r - pad_width
            c_og = c - pad_width
            for i in range(0, 3):
                for j in range(0, 3):
                    x = pad_img[r - 1 + i, c - 1 + j]
                    mu = pad_img[r, c]
                    temp = -(x - mu) ** 2 / (2 * sigma ** 2)
                    g_view[r_og, c_og, i, j] = exp(temp)
            g_view[r_og, c_og, 1, 1] = 0

