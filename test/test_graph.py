import graph
import numpy as np


class TestFullWeightGraph:
    height = 5
    width = 6
    img = np.ones((height, width))

    def test_init(self):
        g = graph.FullWeightedGraph(self.height, self.width)
        assert np.allclose(g.graph, np.ones((self.height, self.width, 3, 3)))

    def test_img_ones(self):
        """
        Should look like:
        1 1 1
        1 N 1
        1 1 1
        at each node
        """
        img = self.img
        g = graph.FullWeightedGraph(self.height, self.width)
        g.set_weights(img, 1)
        ans = np.ones((3, 3))
        ans[1, 1] = 0
        for r in range(0, self.height):
            for c in range(0, self.width):
                assert np.allclose(g.graph[r, c], ans)

    def test_img_1234(self):
        """
        Should look like:
        1  1  a   a  1  1
        1 (1) a - a (2) 1
        b  b  c   a  b  b
           |    X    |
        b  b  a   c  b  b
        1 (3) a - a (4) 1
        1  1  a   a  1  1
        at each node
        """
        a = gaussian(1., 0., 1.)
        b = gaussian(2., 0., 1.)
        c = gaussian(3., 0., 1.)
        img = np.arange(1.0, 5.0).reshape((2, 2))
        g = graph.FullWeightedGraph(2, 2)
        g.set_weights(img, 1)
        ans = np.zeros((2, 2, 3, 3))
        ans[0, 0] = np.array([[1, 1, a],
                              [1, 0, a],
                              [b, b, c]])
        ans[0, 1] = np.array([[a, 1, 1],
                              [a, 0, 1],
                              [a, b, b]])
        ans[1, 0] = np.array([[b, b, a],
                              [1, 0, a],
                              [1, 1, a]])
        ans[1, 1] = np.array([[c, b, b],
                              [a, 0, 1],
                              [a, 1, 1]])
        for r in range(0, 2):
            for c in range(0, 2):
                assert np.allclose(ans[r, c], g.graph[r, c])

    def test_wkernel_ones(self):
        reg_const = 1
        Q = -reg_const ** 2 / 6
        R = - reg_const ** 2 / 12
        w_kernel = np.array([[R, Q, R],
                             [Q, 1, Q],
                             [R, Q, R]])
        ans = w_kernel

        g = graph.FullWeightedGraph(self.height, self.width)
        g.graph = np.ones((self.height, self.width, 3, 3))
        g.apply_w_kernel(reg_const, four_neighbours=False)

        for r in range(0, self.height):
            for c in range(0, self.width):
                assert np.allclose(g.graph[r, c], ans)

    def test_wkernel_4neighbours(self):
        reg_const = 1
        Q = -reg_const
        R = 0
        w_kernel = np.array([[R, Q, R],
                             [Q, 4, Q],
                             [R, Q, R]])

        g = graph.FullWeightedGraph(self.height, self.width)
        g.graph = np.ones((self.height, self.width, 3, 3))
        g.apply_w_kernel(reg_const, four_neighbours=True)

        for r in range(0, self.height):
            for c in range(0, self.width):
                assert np.allclose(g.graph[r, c], w_kernel)

    def test_wkernel_1234(self):
        """
        Should look like:
        1  1  a   a  1  1
        1 (1) a - a (2) 1
        b  b  c   a  b  b
           |    X    |
        b  b  a   c  b  b
        1 (3) a - a (4) 1
        1  1  a   a  1  1
        at each node, but now mulitply by w_kernel
        """
        reg_const = 1
        Q = -reg_const ** 2 / 6
        R = -reg_const ** 2 / 12
        w_kernel = np.array([[R, Q, R],
                             [Q, 0, Q],
                             [R, Q, R]])

        a = gaussian(1., 0., 1.)
        b = gaussian(2., 0., 1.)
        c = gaussian(3., 0., 1.)

        img = np.arange(1.0, 5.0).reshape((2, 2))
        g = graph.FullWeightedGraph(2, 2)
        g.set_weights(img, 1)
        g.apply_w_kernel(1, four_neighbours=False)

        ans = np.zeros((2, 2, 3, 3))
        ans[0, 0] = np.array([[1, 1, a],
                              [1, 0, a],
                              [b, b, c]]) * w_kernel
        ans[0, 0, 1, 1] = -np.sum(ans[0, 0])
        ans[0, 1] = np.array([[a, 1, 1],
                              [a, 0, 1],
                              [a, b, b]]) * w_kernel
        ans[0, 1, 1, 1] = -np.sum(ans[0, 1])
        ans[1, 0] = np.array([[b, b, a],
                              [1, 0, a],
                              [1, 1, a]]) * w_kernel
        ans[1, 0, 1, 1] = -np.sum(ans[1, 0])
        ans[1, 1] = np.array([[c, b, b],
                              [a, 0, 1],
                              [a, 1, 1]]) * w_kernel
        ans[1, 1, 1, 1] = -np.sum(ans[1, 1])
        for r in range(0, 2):
            for c in range(0, 2):
                res = g.graph[r, c]
                assert np.allclose(res, ans[r, c])

    def test_wkernel_1234_4neighbours(self):
        """
        Should look like:
        1  1  a   a  1  1
        1 (1) a - a (2) 1
        b  b  c   a  b  b
           |    X    |
        b  b  a   c  b  b
        1 (3) a - a (4) 1
        1  1  a   a  1  1
        at each node, but now mulitply by w_kernel
        """
        reg_const = 1
        Q = -reg_const
        R = 0
        w_kernel = np.array([[R, Q, R],
                             [Q, 0, Q],
                             [R, Q, R]])

        a = gaussian(1., 0., 1.)
        b = gaussian(2., 0., 1.)
        c = gaussian(3., 0., 1.)

        img = np.arange(1.0, 5.0).reshape((2, 2))
        g = graph.FullWeightedGraph(2, 2)
        g.set_weights(img, 1)
        g.apply_w_kernel(1, True)

        ans = np.zeros((2, 2, 3, 3))
        ans[0, 0] = np.array([[1, 1, a],
                              [1, 0, a],
                              [b, b, c]]) * w_kernel
        ans[0, 0, 1, 1] = -np.sum(ans[0, 0])
        ans[0, 1] = np.array([[a, 1, 1],
                              [a, 0, 1],
                              [a, b, b]]) * w_kernel
        ans[0, 1, 1, 1] = -np.sum(ans[0, 1])
        ans[1, 0] = np.array([[b, b, a],
                              [1, 0, a],
                              [1, 1, a]]) * w_kernel
        ans[1, 0, 1, 1] = -np.sum(ans[1, 0])
        ans[1, 1] = np.array([[c, b, b],
                              [a, 0, 1],
                              [a, 1, 1]]) * w_kernel
        ans[1, 1, 1, 1] = -np.sum(ans[1, 1])
        for r in range(0, 2):
            for c in range(0, 2):
                res = g.graph[r, c]
                assert np.allclose(res, ans[r, c])

    def test_graph_sum_ones(self):
        img = self.img
        ans = 0
        g = graph.FullWeightedGraph(self.height, self.width)
        g.set_weights(img, 1)
        g.apply_w_kernel(1)
        assert np.allclose(g.graph_sum(img), ans)

    def test_graph_sum_inplace_ones(self):
        img = self.img
        ans = 0
        g = graph.FullWeightedGraph(self.height, self.width)
        g.set_weights(img, 1)
        g.apply_w_kernel(1)
        res = np.empty_like(img)
        g.g_sum_inplace(img, res)
        assert np.allclose(res, ans)

    def test_graph_sum_1234(self):
        reg_const = 1
        Q = -reg_const ** 2 / 6
        R = -reg_const ** 2 / 12
        w_kernel = np.array([[R, Q, R],
                             [Q, 0, Q],
                             [R, Q, R]])

        img = np.arange(1.0, 5.0).reshape((2, 2))
        g = graph.FullWeightedGraph(2, 2)
        g.set_weights(img, 1)
        g.apply_w_kernel(1, four_neighbours=False)

        a = gaussian(1., 0., 1.)
        b = gaussian(2., 0., 1.)
        c = gaussian(3., 0., 1.)

        ans = np.zeros((2, 2, 3, 3))
        ans[0, 0] = np.array([[1, 1, a],
                              [1, 0, a],
                              [b, b, c]]) * w_kernel
        ans[0, 0, 1, 1] = -np.sum(ans[0, 0])
        ans[0, 1] = np.array([[a, 1, 1],
                              [a, 0, 1],
                              [a, b, b]]) * w_kernel
        ans[0, 1, 1, 1] = -np.sum(ans[0, 1])
        ans[1, 0] = np.array([[b, b, a],
                              [1, 0, a],
                              [1, 1, a]]) * w_kernel
        ans[1, 0, 1, 1] = -np.sum(ans[1, 0])
        ans[1, 1] = np.array([[c, b, b],
                              [a, 0, 1],
                              [a, 1, 1]]) * w_kernel
        ans[1, 1, 1, 1] = -np.sum(ans[1, 1])

        pad_img = np.array([[1., 1., 2., 2.],
                            [1., 1., 2., 2.],
                            [3., 3., 4., 4.],
                            [3., 3., 4., 4.]])

        g_sum = np.zeros((2, 2))
        g_sum[0, 0] = np.sum(pad_img[0:3, 0:3] * ans[0, 0])
        g_sum[0, 1] = np.sum(pad_img[0:3, 1:4] * ans[0, 1])
        g_sum[1, 0] = np.sum(pad_img[1:4, 0:3] * ans[1, 0])
        g_sum[1, 1] = np.sum(pad_img[1:4, 1:4] * ans[1, 1])

        res = g.graph_sum(img)
        assert np.allclose(res, g_sum)

    def test_graph_sum_1234_inplace(self):
        reg_const = 1
        Q = -reg_const ** 2 / 6
        R = -reg_const ** 2 / 12
        w_kernel = np.array([[R, Q, R],
                             [Q, 0, Q],
                             [R, Q, R]])

        img = np.arange(1.0, 5.0).reshape((2, 2))
        g = graph.FullWeightedGraph(2, 2)
        g.set_weights(img, 1)
        g.apply_w_kernel(1, four_neighbours=False)

        a = gaussian(1., 0., 1.)
        b = gaussian(2., 0., 1.)
        c = gaussian(3., 0., 1.)

        ans = np.zeros((2, 2, 3, 3))
        ans[0, 0] = np.array([[1, 1, a],
                              [1, 0, a],
                              [b, b, c]]) * w_kernel
        ans[0, 0, 1, 1] = -np.sum(ans[0, 0])
        ans[0, 1] = np.array([[a, 1, 1],
                              [a, 0, 1],
                              [a, b, b]]) * w_kernel
        ans[0, 1, 1, 1] = -np.sum(ans[0, 1])
        ans[1, 0] = np.array([[b, b, a],
                              [1, 0, a],
                              [1, 1, a]]) * w_kernel
        ans[1, 0, 1, 1] = -np.sum(ans[1, 0])
        ans[1, 1] = np.array([[c, b, b],
                              [a, 0, 1],
                              [a, 1, 1]]) * w_kernel
        ans[1, 1, 1, 1] = -np.sum(ans[1, 1])

        pad_img = np.array([[1., 1., 2., 2.],
                            [1., 1., 2., 2.],
                            [3., 3., 4., 4.],
                            [3., 3., 4., 4.]])

        g_sum = np.zeros((2, 2))
        g_sum[0, 0] = np.sum(pad_img[0:3, 0:3] * ans[0, 0])
        g_sum[0, 1] = np.sum(pad_img[0:3, 1:4] * ans[0, 1])
        g_sum[1, 0] = np.sum(pad_img[1:4, 0:3] * ans[1, 0])
        g_sum[1, 1] = np.sum(pad_img[1:4, 1:4] * ans[1, 1])
        res = np.empty_like(img)
        g.g_sum_inplace(img, res)
        assert np.allclose(res, g_sum)


def gaussian(x, mu, sig):
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def test_set_l1_weights_4_neighbours_flat():
    pad_w = np.ones((7, 7))
    res = np.zeros((5, 5))
    delta = 1e-7
    height = 5
    width = 5
    graph.debug_agg_l1_weights_4_neighbours(pad_w, res, height, width, 1)
    assert np.allclose(1 / (2 * np.sqrt(delta)), res)


def test_set_l1_weights_4_neighbours_y_grad():
    res = np.zeros((3, 3))
    pad_w = np.array([[1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2],
                      [3, 3, 3, 3, 3],
                      [4, 4, 4, 4, 4],
                      [5, 5, 5, 5, 5]], dtype=float)
    height = 3
    width = 3
    delta = 1e-7
    graph.debug_agg_l1_weights_4_neighbours(pad_w, res, height, width, 1)
    print(res)
    ans = 1 / (2 * np.sqrt(1 + delta))
    assert np.allclose(ans, res)


def test_set_l1_weights_4_neighbours_x_grad():
    res = np.zeros((3, 3))
    pad_w = np.array([[1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2],
                      [3, 3, 3, 3, 3],
                      [4, 4, 4, 4, 4],
                      [5, 5, 5, 5, 5]], dtype=float)
    pad_w = np.rot90(pad_w)
    height = 3
    width = 3
    delta = 1e-8
    graph.debug_agg_l1_weights_4_neighbours(pad_w, res, height, width, 1)
    ans = 1 / (2 * np.sqrt(1 + delta))
    assert np.allclose(ans, res)


def test_set_l1_weights_4_neighbours_xy_grad():
    res = np.zeros((3, 3))
    x_grad = np.array([[1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]], dtype=float)
    y_grad = np.rot90(x_grad)
    pad_w = x_grad + y_grad
    height = 3
    width = 3
    delta = 1e-8
    graph.debug_agg_l1_weights_4_neighbours(pad_w, res, height, width, 1)
    ans = 1 / (2 * np.sqrt(2 + delta))
    assert np.allclose(ans, res)


class TestSet4NeighboursGraphWeights:
    h = 3
    w = 3
    pad_size = 1
    pad_h = h + 2
    pad_w = w + 2

    def test_flat(self):
        pad_weights = np.ones((self.pad_h, self.pad_w))
        g_res = np.zeros((self.h, self.w, 3, 3))
        graph.debug_set_4_neighbours_graph_weights(pad_weights, g_res, self.h,
                                                   self.w, self.pad_size, 0)
        helper_assert_corners_zero(g_res)

        for r in range(self.h):
            for c in range(self.w):
                # assert that the sides are all ones.
                assert g_res[r, c, 0, 1] == 1
                assert g_res[r, c, 1, 2] == 1
                assert g_res[r, c, 2, 1] == 1
                assert g_res[r, c, 1, 0] == 1

    def test_impulse(self):
        pad_weights = np.zeros((self.pad_h, self.pad_w))
        pad_weights[2, 2] = 1.0
        g_res = np.zeros((self.h, self.w, 3, 3))
        graph.debug_set_4_neighbours_graph_weights(pad_weights, g_res, self.h,
                                                   self.w, self.pad_size, 0)
        helper_assert_corners_zero(g_res)

        for r in range(self.h):
            for c in range(self.w):
                if r == 1 and c == 1:
                    assert g_res[r, c, 0, 1] == 1 / 2
                    assert g_res[r, c, 1, 2] == 1 / 2
                    assert g_res[r, c, 2, 1] == 1 / 2
                    assert g_res[r, c, 1, 0] == 1 / 2
                elif r == 1 and c == 0:
                    assert g_res[r, c, 0, 1] == 0
                    assert g_res[r, c, 1, 2] == 1 / 2
                    assert g_res[r, c, 2, 1] == 0
                    assert g_res[r, c, 1, 0] == 0
                elif r == 1 and c == 2:
                    assert g_res[r, c, 0, 1] == 0
                    assert g_res[r, c, 1, 2] == 0
                    assert g_res[r, c, 2, 1] == 0
                    assert g_res[r, c, 1, 0] == 1 / 2
                elif r == 0 and c == 1:
                    assert g_res[r, c, 0, 1] == 0
                    assert g_res[r, c, 1, 2] == 0
                    assert g_res[r, c, 2, 1] == 1 / 2
                    assert g_res[r, c, 1, 0] == 0
                elif r == 2 and c == 1:
                    assert g_res[r, c, 0, 1] == 1 / 2
                    assert g_res[r, c, 1, 2] == 0
                    assert g_res[r, c, 2, 1] == 0
                    assert g_res[r, c, 1, 0] == 0
                else:
                    for i in range(3):
                        for j in range(3):
                            assert g_res[r, c, i, j] == 0


def test_normalise():
    from scipy import ndimage
    height = 4
    width = 4
    g = graph.FullWeightedGraph(height, width)
    g.apply_w_kernel(-1.0)
    g.graph[:, :, 1, 1] = 0
    d_weights = np.array([[[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]]]], dtype=float)
    pos_arr = np.array([0.5, 0.5, 0])
    neg_arr = np.array([0., 0.5, 0.5])
    g_r = ndimage.convolve1d(d_weights[0, 0], pos_arr)
    g_l = ndimage.convolve1d(d_weights[0, 0], neg_arr)
    g_d = ndimage.convolve1d(d_weights[0, 0], pos_arr, axis=0)
    g_u = ndimage.convolve1d(d_weights[0, 0], neg_arr, axis=0)
    g.normalise(d_weights)
    helper_assert_corners_zero(g.graph)
    assert np.allclose(g.graph[:, :, 0, 1], g_u)
    assert np.allclose(g.graph[:, :, 2, 1], g_d)
    assert np.allclose(g.graph[:, :, 1, 0], g_l)
    assert np.allclose(g.graph[:, :, 1, 2], g_r)



def helper_assert_corners_zero(g_res):
    for r in range(g_res.shape[0]):
        for c in range(g_res.shape[1]):
            assert g_res[r, c, 0, 0] == 0
            assert g_res[r, c, 0, 2] == 0
            assert g_res[r, c, 2, 0] == 0
            assert g_res[r, c, 2, 2] == 0
