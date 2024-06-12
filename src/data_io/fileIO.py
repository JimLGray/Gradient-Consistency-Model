# Module fileIO.py

from PIL import Image
import numpy as np
import re
import sys
import h5py
from data_io import plot
from scipy import io


def open_img(file: str) -> np.ndarray:
    """
    Opens image using Pillow converts it to greyscale and returns it
    as a numpy array.

    Parameters
    ----------
    file

    Returns
    -------

    """
    if file[-3:] == 'npy':
        img = np.load(file)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        grey = (R * 0.299 + G * 0.587 + B * 0.114) / 255
    else:
        img = Image.open(file).convert("L")
        grey = np.array(img) / 255
    return grey

def readPFM(file):
    """
    Modified from
    https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py

    pfm doc is taken from here:
    http://netpbm.sourceforge.net/doc/pfm.html

    Parameters
    ----------
    file

    Returns
    -------

    """
    with open(file, 'rb') as f:

        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_line = f.readline()
        dim_str = dim_line.decode('ascii')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data, scale


def writePFM(filename, image, scale=1):
    """
    Modified from
    https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py

    pfm doc is taken from here:
    http://netpbm.sourceforge.net/doc/pfm.html

    Parameters
    ----------
    filename
    image
    scale

    Returns
    -------

    """
    with open(filename, 'wb') as f:

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[
            2] == 1:  # greyscale
            color = False
        else:
            raise Exception(
                'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        if color:
            identifier_str = 'PF\n'
        else:
            identifier_str = 'Pf\n'
        identifier_line = bytes(identifier_str, 'ascii')
        f.write(identifier_line)

        dimension_str = '%d %d\n' % (image.shape[1], image.shape[0])
        dimension_line = bytes(dimension_str, 'ascii')
        f.write(dimension_line)

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        scale_str = '%f\n' % scale
        scale_line = bytes(scale_str, 'ascii')
        f.write(scale_line)

        image.tofile(f)


def read_png_disp(png: str, scale=1.0):
    """
    Reads data from a .png disparity file. We assume that the disparity values
    are as is, up to a scale and that zero disparity means NAN. This obviously
    doesn't hold for all files.

    This was written to deal with the spec of the middlebury stereo datasets.

    Parameters
    ----------
    png: str
        name of .png file
    scale: float
        the size of the scaling factor to apply to disp

    Returns
    -------

    """
    img = Image.open(png)
    disp = np.array(img) * scale
    disp = np.where(disp == 0, np.nan, disp)
    return disp


def read_vmin_vmax(cfg):
    """
    Finds the min and max disparity as per the .cfg file. Assumes that the there
    will be a line in the file that says "disp_min = -2.9" for example and one
    that has "disp_max = 1.2".

    Parameters
    ----------
    cfg: str
        Filename for .cfg file

    Returns
    -------
    vmin, vmax
        floats with minimum disparity and the maximum disparity as per the file.
    """

    min_pattern = '\ndisp_min = .*\n'
    max_pattern = '\ndisp_max = .*\n'
    with open(cfg) as f:
        read_data = f.read()

    min_match = re.search(min_pattern, read_data)
    max_match = re.search(max_pattern, read_data)

    if min_match is None:
        raise ValueError('No match found for min')
    if max_match is None:
        raise ValueError('No match found for max')

    min_str = min_match[0]
    max_str = max_match[0]

    # remove "\ndisp_min =" and remove the newline then convert to float
    min_val = float(min_str[12:-1])
    max_val = float(max_str[12:-1])
    return min_val, max_val


def open_middlebury2006_gt(file: str, scale=0.5):

    view_diff = 4
    img = Image.open(file)
    grey = np.array(img) * scale / view_diff
    gt = np.where(grey == 0, np.nan, grey)
    return gt


def open_hci_2013(file: str, dst: str):
    """
    See
    http://publications.lightfield-analysis.net/WMG13_vmv.pdf

    We open the file and then conver the ground truth to disparity. We also get
    the image data and the baselines.

    Converts depth to disparity using the equation
     d = (B * f) / Z − ∆x

    Parameters
    ----------
    file
    dst: str
        name of the folder to save the images and the groundtruth data.

    """
    f = h5py.File(file)
    lf = np.array(f['LF'])
    mat_name = dst + 'lf.mat'
    write_lf_mat(lf, mat_name)

    max_x = lf.shape[0]
    max_y = lf.shape[1]
    for x in range(max_x):
        for y in range(max_y):
            true_x = -int(x - (max_x - 1)/2)
            true_y = -int(y - (max_y - 1)/2)
            # img = Image.fromarray(lf[y, x])
            # dst_name = dst + '(' + str(true_x) + ', ' + str(true_y) + ').png'
            dst_name = dst + '(' + str(true_x) + ', ' + str(true_y) + ').npy'
            img = lf[y, x]
            np.save(dst_name, lf[y, x])

    gt_depth = f['GT_DEPTH']
    gt_depth_ref = gt_depth[4, 4]
    # Now to convert to disparity.
    B = f.attrs['dH'][0]
    focal_length = f.attrs['focalLength'][0]
    delta_x = f.attrs['shift']
    gt_disparity = (B * focal_length) / gt_depth_ref - delta_x
    gt_name = dst + 'gt_disp_lowres.pfm'
    writePFM(gt_name, gt_disparity)

    gt_depth_name = dst + 'gt_depth_lowres.pfm'
    writePFM(gt_depth_name, gt_depth_ref)


def write_lf_mat(lf, f_name: str):
    """
    writes the light field as a .mat file, so that matlab can read it.
    Parameters
    ----------
    lf:
        5D array. [y, x, row, col, channel]
    f_name: str
        filename to be written

    Returns
    -------

    """

    mdict = {'LF': np.flip(lf, 1)}
    io.savemat(f_name, mdict)
