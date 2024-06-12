# distutils: language = c++

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef  np.float64_t DTYPE_t

ctypedef np.npy_float32 flt32

from cython.operator cimport dereference as deref
from cython cimport view

cdef extern from 'map_depth.h':
    struct depth_info:
        pass

cdef extern from "frame.h" namespace "sd":
    cdef cppclass inter_frame_flt32:
        inter_frame_flt32() except +
        inter_frame_flt32(int width, int height, int num_channels,
                          int extension, flt32* buf) except +
        flt32* get_row(int row)
        int get_width()
        int get_height()
        int get_stride()
        int get_num_ch()

    cdef cppclass inter_frame_info:
        inter_frame_info() except+

    struct floc:
        flt32 x
        flt32 y

cdef extern from "map_depth.cpp":
    void reverse_map_image(
            inter_frame_flt32* tgt,
            inter_frame_flt32* disparity,
            inter_frame_flt32* out,
            inter_frame_info* tmp,
            inter_frame_flt32* acc,
            const floc odisp,
            const int dsubsam,
            const float max_expansion
    )

