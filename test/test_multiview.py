import numpy as np
import multiview as mv
from img import Images


class TestLocalLimit:
    h = 5
    w = 5
    n_imgs = 17
    max_q = 3
    coeffs = np.ones((n_imgs + 1, max_q, h, w))
    baselines = np.array([(1, 0)] * 18)
    imgs = Images.set(coeffs, baselines)
    grads = np.ones((n_imgs, max_q, h, w))

    def test_all_top_level(self):
        W_pq = np.zeros((self.n_imgs, self.max_q, self.h, self.w))
        q = 0
        W_pq[:, q] = 1.0
        res = mv.local_limit(W_pq, self.grads, self.imgs)
        sig = np.sqrt(2) * 2. ** float(q) / 2.
        ans = np.ones((self.h, self.w)) * 2 * np.sqrt(2) * sig
        assert np.allclose(res, ans)

    def test_all_bottom_level(self):
        W_pq = np.zeros((self.n_imgs, self.max_q, self.h, self.w))
        q = self.max_q - 1
        W_pq[:, q] = 1.0
        res = mv.local_limit(W_pq, self.grads, self.imgs)
        sig = np.sqrt(2) * 2. ** float(q) / 2.
        ans = np.ones((self.h, self.w)) * 2 * np.sqrt(2) * sig
        assert np.allclose(res, ans)

    def test_middle_level(self):
        W_pq = np.zeros((self.n_imgs, self.max_q, self.h, self.w))
        q = 1
        W_pq[:, q] = 1.0
        res = mv.local_limit(W_pq, self.grads, self.imgs)
        sig = np.sqrt(2) * 2. ** float(q) / 2.
        ans = np.ones((self.h, self.w)) * 2 * np.sqrt(2) * sig
        assert np.allclose(res, ans)

    def test_all_equal(self):
        W_pq = np.ones((self.n_imgs, self.max_q, self.h, self.w))
        sig = 7.0 * np.sqrt(2) / 6.0
        ans = np.ones((self.h, self.w)) * 2 * np.sqrt(2) * sig
        res = mv.local_limit(W_pq, self.grads, self.imgs)
        assert np.allclose(res, ans)
