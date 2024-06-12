import numpy as np
cimport numpy as np


DTYPE = np.float64
ctypedef  np.float64_t DTYPE_t


cdef filter_sep2d(np.ndarray[DTYPE_t, ndim=2] img,
                  np.ndarray[DTYPE_t, ndim=1] kernel,
                  str mode, float cval)

cdef filter1d(np.ndarray[DTYPE_t, ndim=2] img,
              np.ndarray[DTYPE_t, ndim=1] kernel,
              int axis, str mode, float cval)

cdef  np.ndarray[DTYPE_t, ndim=2] filter2d(np.ndarray[DTYPE_t, ndim=2] img,
                                           np.ndarray[DTYPE_t, ndim=2] kernel,
                                           str mode, float cval)