import numpy as np
import data_io.fileIO as gt
import os
import warp


im = np.random.rand(10, 10)
disp = np.ones((10, 10))
b_x = 1.0
b_y = 1.0
out = warp.reverse_map_wrapper(im, disp, b_x, b_y)

im_name = 'tmp/im.pfm'
disp_name = 'tmp/disp.pfm'
out_name = 'tmp/out.pfm'
gt.writePFM(im_name, im.astype(np.float32))
gt.writePFM(disp_name, disp.astype(np.float32))
focal_length = 1

cmd_args = [
    'tmp/warp_lib_cli',
    im_name,
    disp_name,
    out_name,
    '-b',
    '1',
    str(focal_length),
    str(b_x),
    str(b_y)
]
cmd_str = ' '.join(cmd_args)
# join all the arguments with a space between them
os.system(cmd_str)
ans, _ = gt.readPFM(out_name)
assert np.allclose(out, ans)

