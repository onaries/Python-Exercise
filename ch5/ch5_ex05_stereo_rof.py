from PIL import Image
import numpy
import rof
import stereo

im_l = numpy.array(Image.open('./data/ppm/scene1.row3.col3.ppm').convert('L'), 'f')
im_r = numpy.array(Image.open('./data/ppm/scene1.row3.col4.ppm').convert('L'), 'f')

steps = 12
start = 4
wid = 9

res = stereo.plane_sweep_ncc(im_l, im_r, start, steps, wid)

res, _ = rof.denoise(res, res, tv_weight=80/255.0, tolerance=0.01)

import scipy.misc
scipy.misc.imsave('./output/depth_rof.png', res)
