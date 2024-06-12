import cgrad as cg
import numpy as np
import graph


class TestTenXTen:
    height = 10
    width = 10

    def test_cg_disparity_random(self):
        grads = np.random.random((self.height, self.width))
        inter = np.zeros_like(grads)

        reg = graph.FullWeightedGraph(self.height, self.width)

        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1)
        ans = np.zeros_like(grads)
        assert np.allclose(res, ans)

    def test_cg_disparity_data_ones(self):
        grads = np.ones((self.height, self.width))
        inter = np.ones_like(grads)

        reg = graph.FullWeightedGraph(self.height, self.width)
        reg.graph = np.zeros_like(reg.graph)
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1)
        ans = np.ones_like(grads)
        assert np.allclose(res, ans)

    def test_rand_data_only(self):
        grads = np.random.rand(self.height, self.width) + 1.0
        inter = np.random.rand(self.height, self.width) + 1.0

        reg = graph.FullWeightedGraph(self.height, self.width)
        reg.graph = np.zeros_like(reg.graph)
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1000, tol=1e-8)
        ans = np.reciprocal(grads) * inter
        assert np.allclose(res, ans)

    # def test_speed(self):
    #     grads = np.random.random((self.height, self.width))
    #     inter = np.random.random((self.height, self.width))
    #     g_data = gd.GradData.set(grads, inter)
    #
    #     reg = graph.FullWeightedGraph(self.height, self.width)
    #     reg.graph = np.zeros_like(reg.graph)
    #     disparity = np.zeros_like(grads)
    #
    #     res = cg_disparity(disparity, g_data, reg, 1000000)


class TestFourXFour:
    height = 4
    width = 4

    def test_ones_data_reg_ones(self):
        grads = np.ones((self.height, self.width)) * 2.0
        inter = np.ones((self.height, self.width))
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        # ans = np.reciprocal(grads) * inter
        d = grads.reshape(grads.size)
        A_data = np.diag(d)
        # it's all to do with boundary conditions...
        reg = graph_helper(self.height, self.width, 1, 1)
        C = np.array([[1, 1, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [0, 0, 1, 1]])
        z = np.zeros_like(C)
        A_reg = np.block([[C, C, z, z],
                          [C, C, C, z],
                          [z, C, C, C],
                          [z, z, C, C]])
        A = A_reg + A_data
        A_inv = np.linalg.inv(A)
        b = inter.reshape(inter.size)
        ans = A_inv @ b

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1000, tol=1e-8)
        res_flat = res.reshape(res.size)
        t = A @ res_flat
        assert np.allclose(t, b)
        assert np.allclose(res_flat, ans)

    def test_rand_data_reg_ones(self):
        grads = np.random.rand(self.height, self.width) + 1.0
        inter = np.random.rand(self.height, self.width) + 1.0
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        # ans = np.reciprocal(grads) * inter
        d = grads.reshape(grads.size)
        A_data = np.diag(d)
        # it's all to do with boundary conditions...
        reg = graph_helper(self.height, self.width, 1, 1)
        C = np.array([[1, 1, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [0, 0, 1, 1]])
        z = np.zeros_like(C)
        A_reg = np.block([[C, C, z, z],
                          [C, C, C, z],
                          [z, C, C, C],
                          [z, z, C, C]])
        A = A_reg + A_data
        A_inv = np.linalg.inv(A)
        b = inter.reshape(inter.size)
        ans = A_inv @ b

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1000, tol=1e-8)
        res_flat = res.reshape(res.size)
        t = A @ res_flat
        assert np.allclose(t, b)
        assert np.allclose(res_flat, ans)

    def test_neg_reg(self):
        grads = np.ones((self.height, self.width)) * 50.0
        inter = np.ones((self.height, self.width))
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        # ans = np.reciprocal(grads) * inter
        d = grads.reshape(grads.size)
        A_data = np.diag(d)
        # it's all to do with boundary conditions...
        reg = graph_helper(self.height, self.width, -1, -1)
        C = -np.array([[1, 1, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [0, 0, 1, 1]])
        z = np.zeros_like(C)
        A_reg = np.block([[C, C, z, z],
                          [C, C, C, z],
                          [z, C, C, C],
                          [z, z, C, C]])
        A = A_reg + A_data
        A_inv = np.linalg.inv(A)
        b = inter.reshape(inter.size)
        ans = A_inv @ b

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1000, tol=1e-8)
        res_flat = res.reshape(res.size)
        t = A @ res_flat
        assert np.allclose(t, b)
        assert np.allclose(res_flat, ans)

    def test_neg_reg(self):
        grads = np.ones((self.height, self.width)) * 50.0
        inter = np.ones((self.height, self.width))
        dw = np.zeros_like(grads)
        w = np.zeros_like(grads)

        # ans = np.reciprocal(grads) * inter
        d = grads.reshape(grads.size)
        A_data = np.diag(d)
        # it's all to do with boundary conditions...
        reg = graph_helper(self.height, self.width, -1, -1)
        C = -np.array([[1, 1, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [0, 0, 1, 1]])
        z = np.zeros_like(C)
        A_reg = np.block([[C, C, z, z],
                          [C, C, C, z],
                          [z, C, C, C],
                          [z, z, C, C]])
        A = A_reg + A_data
        A_inv = np.linalg.inv(A)
        b = inter.reshape(inter.size)
        ans = A_inv @ b

        res = cg.cg_disparity(w, dw, grads, inter, reg, 1000, tol=1e-8)
        res_flat = res.reshape(res.size)
        t = A @ res_flat
        assert np.allclose(t, b)
        assert np.allclose(res_flat, ans)


def graph_helper(height, width, vertex, centre):
    reg = graph.FullWeightedGraph(height, width)
    for r in range(height):
        for c in range(width):
            if (0 < r < height - 1) and (0 < c < width - 1):
                # middle
                reg.graph[r, c] = np.array([[vertex, vertex, vertex],
                                            [vertex, centre, vertex],
                                            [vertex, vertex, vertex]])
            elif r == 0 and (0 < c < width - 1):
                # top edge
                reg.graph[r, c] = np.array([[0, 0, 0],
                                            [vertex, centre, vertex],
                                            [vertex, vertex, vertex]])
            elif r == 0 and c == 0:
                # top left corner
                reg.graph[r, c] = np.array([[0, 0, 0],
                                            [0, centre, vertex],
                                            [0, vertex, vertex]])
            elif r == 0 and c == width - 1:
                # top right corner
                reg.graph[r, c] = np.array([[0, 0, 0],
                                            [vertex, centre, 0],
                                            [vertex, vertex, 0]])
            elif r == height - 1 and (0 < c < width - 1):
                # bottom edge
                reg.graph[r, c] = np.array([[vertex, centre, vertex],
                                            [vertex, vertex, vertex],
                                            [0, 0, 0]])
            elif r == height - 1 and c == 0:
                # bottom left corner
                reg.graph[r, c] = np.array([[0, centre, vertex],
                                            [0, vertex, vertex],
                                            [0, 0, 0]])
            elif r == height - 1 and c == width - 1:
                # bottom right corner
                reg.graph[r, c] = np.array([[vertex, centre, 0],
                                            [vertex, vertex, 0],
                                            [0, 0, 0]])
            elif c == 0:
                # left edge
                reg.graph[r, c] = np.array([[0, vertex, vertex],
                                            [0, centre, vertex],
                                            [0, vertex, vertex]])
            elif c == width - 1:
                # right edge
                reg.graph[r, c] = np.array([[vertex, vertex, 0],
                                            [vertex, centre, 0],
                                            [vertex, vertex, 0]])
    return reg
