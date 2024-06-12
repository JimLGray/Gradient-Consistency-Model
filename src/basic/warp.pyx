import numpy as np
cimport cython
import cv2 as cv
from libc.math cimport sqrt, ceil, floor, round


def fast_warp(img, flows, mode='cubic', bordermode='reflect'):
    # This actually warps backward to the flows
    end_points = flow_endpoints(flows)
    map_x = end_points[:, :, 0].astype(np.float32)
    map_y = end_points[:, :, 1].astype(np.float32)
    if mode == 'cubic' or mode == 'bicubic':
        f_mode = cv.INTER_CUBIC
    elif mode == 'linear' or mode == 'bilinear':
        f_mode = cv.INTER_LINEAR

    if bordermode == 'reflect':
        b_mode = cv.BORDER_REFLECT
    elif bordermode == 'edge' or bordermode == 'nearest':
        b_mode = cv.BORDER_REPLICATE

    dst = cv.remap(img, map_x, map_y, f_mode, b_mode)
    return dst


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_t, ndim=2] bilinear_warp(
        np.ndarray[DTYPE_t, ndim=2] ref_img,
        np.ndarray[DTYPE_t, ndim=2] tgt_img,
        np.ndarray[DTYPE_t, ndim=3] flows
):
    """
    Performs warping using bilinear interpolation. Also if the flows points
    somewhere oustide the image boundary then we use the reference image.

    Interpolation bit is based on:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Parameters
    ----------
    ref_img
    tgt_img
    flows

    Returns
    -------

    """
    cdef double[:, :, :] v = flows
    cdef int height = flows.shape[0]
    cdef int width = flows.shape[1]
    cdef int r, c,
    pad_tgt = np.pad(tgt_img, 1, mode='reflect')
    cdef double[:, :] tgt = pad_tgt
    output = np.empty_like(tgt_img)
    cdef double[:, :] out = output
    cdef double[:, :] ref = ref_img

    cdef double v_x, v_y, fv_x, fv_y, w_11, w_12, w_21, w_22
    cdef int vi_x, vi_y
    cdef double f_00, f_10, f_01, f_11
    for r in range(height):
        for c in range(width):
            v_x = v[r, c, 0] + c + 1 # add pad width
            v_y = v[r, c, 1] + r + 1 # add pad_width
            # if (0 < v_x < width + 1) and (0 < v_y < height + 1):
            # it seems as if their method for some reason cuts off 1 early.
            if (1 < v_x < width) and (1 < v_y < height):
                vi_x = int(floor(v_x))
                vi_y = int(floor(v_y))
                fv_x = v_x - vi_x
                fv_y = v_y - vi_y

                w_11 = (1 - fv_x) * (1 - fv_y)
                w_12 = (1 - fv_x) * fv_y
                w_21 = fv_x * (1 - fv_y)
                w_22 = fv_y * fv_x

                f_00 = tgt[vi_y, vi_x]
                f_10 = tgt[vi_y, vi_x + 1]
                f_01 = tgt[vi_y + 1, vi_x]
                f_11 = tgt[vi_y + 1, vi_x + 1]
                out[r, c] = w_11*f_00 + w_12*f_01 + w_21*f_10 + w_22*f_11

            else:
                out[r, c] = ref[r, c]
    return output


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
def flow_endpoints(double[:, :, :] flows):
    """
    Calculates where all the pixels would end up, if they were moved according
    to their flows. Double mappings are not considered here, neither is
    rounding, nor image boundaries.

    :param flows: a 3D array of dimensions (height, width, 2), which at each x,y
        location in the image contains a (u,v) vector with the flow's horizontal
        and vertical components
    :return: an array containing the endpoint coordinates of the flows, in
        (x_final, y_final) form
    """
    endpoints = np.empty_like(flows)
    cdef double[:, :, :] endpoints_view = endpoints
    cdef double[:, :, :] flows_view = flows
    cdef int height = flows.shape[0]
    cdef int width = flows.shape[1]
    cdef int i, j
    cdef double x_flow, y_flow
    for i in range(0, height):
        for j in range(0, width):
            x_flow = flows_view[i, j, 0]
            y_flow = flows_view[i, j, 1]
            endpoints_view[i, j, 0] = j + x_flow
            endpoints_view[i, j, 1] = i + y_flow

    return endpoints


cpdef np.ndarray[DTYPE_t, ndim=2] reverse_map_wrapper(
        np.ndarray[DTYPE_t, ndim=2] im,
        np.ndarray[DTYPE_t, ndim=2] disp,
        float b_x,
        float b_y
):
    """
    wraps the warp_lib_cli stuff. occluded areas are marked zero.

    Parameters
    ----------
    im:
        image to warp
    disp:
        disparity
    b_x: float
        baseline in the x direction
    b_y: float
        baseline in the y direction

    Returns
    -------

    """
    cdef int height = im.shape[0]
    cdef int width = im.shape[1]
    cdef int n_chan = 1
    cdef int b_ext = 16
    im32 = np.ascontiguousarray(im, dtype=np.single)
    disp32 = np.ascontiguousarray(disp, dtype=np.single)
    im_holder = np.empty((height + 2 * b_ext, width + 2 * b_ext))
    d_holder = np.empty((height + 2 * b_ext, width + 2 * b_ext))
    im_cont = np.ascontiguousarray(im_holder, dtype=np.single)
    d_cont = np.ascontiguousarray(d_holder, dtype=np.single)

    out = np.empty_like(im_holder)
    o_cont = np.ascontiguousarray(out, dtype=np.single)
    res = np.empty_like(im, dtype=np.single)

    cdef flt32[:, ::1] im_view = im_cont
    cdef flt32[:, ::1] disp_view = d_cont
    cdef flt32[:, ::1] out_view = o_cont
    cdef flt32[:, ::1] res_view = res

    cdef inter_frame_flt32 im_frame = inter_frame_flt32(width, height, n_chan,
                                                        b_ext, &im_view[0][0])
    cdef inter_frame_flt32 disp_frame = inter_frame_flt32(width, height, n_chan,
                                                          b_ext,
                                                          &disp_view[0][0])
    cdef inter_frame_flt32 o_frame = inter_frame_flt32(width, height, n_chan,
                                                       b_ext, &out_view[0][0])
    populate_frame(im_frame, im32)
    populate_frame(disp_frame, disp32)
    cdef inter_frame_flt32 acc = inter_frame_flt32()
    cdef inter_frame_info tmp = inter_frame_info()

    cdef floc baseline
    baseline.x = b_x
    baseline.y = b_y

    reverse_map_image(
        &im_frame,
        &disp_frame,
        &o_frame,
        &tmp,
        &acc,
        baseline,
        1,
        2.0
    )
    get_frame_data(o_frame, res_view)

    return res


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cdef void populate_frame(inter_frame_flt32 frame,
                         flt32[:, :] data):
    """
    Filles frame with the contents of data.
    
    Parameters
    ----------
    frame
    data

    Returns
    -------

    """
    cdef int height = frame.get_height()
    cdef int width = frame.get_width()
    cdef int stride = frame.get_stride()
    if data.shape[0] != height:
        raise ValueError('Data height: ' + str(data.shape[0])  +
                         ' and frame height: ' + str(height) + ' do not match')
    if data.shape[1] != width:
        raise ValueError('Data width: ' + str(data.shape[1])  +
                         ' and frame width: ' + str(width) + ' do not match')

    cdef int r, c, idx
    cdef flt32* dptr

    dptr = frame.get_row(0)
    for r in range(height):
        for c in range(width):
            idx = r * stride + c
            dptr[idx] = data[r, c]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cdef void get_frame_data(inter_frame_flt32 frame, flt32[:, :] dest):
    cdef int height = frame.get_height()
    cdef int width = frame.get_width()
    cdef int stride = frame.get_stride()
    if dest.shape[0] != height:
        raise ValueError('Data height: ' + str(dest.shape[0])  +
                         ' and frame height: ' + str(height) + ' do not match')
    if dest.shape[1] != width:
        raise ValueError('Data width: ' + str(dest.shape[1])  +
                         ' and frame width: ' + str(width) + ' do not match')

    cdef int r, c, idx
    cdef flt32* dptr

    dptr = frame.get_row(0)
    for r in range(height):
        for c in range(width):
            idx = r * stride + c
            dest[r, c] = dptr[idx]

