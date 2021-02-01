## convert npy to image
'''
date:   2018-11-13
author: kuo
'''
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import os
from PIL import Image
import PIL.Image as pil

postproc = True
if postproc:
    surfix = '_pp'
else:
    surfix = ''
## setup root path
path2root   = '../../models/model_cityscapes_resnet/' #20_regh/'
out_dir = 'make3d{}/'.format(surfix)
path2out = path2root + out_dir 
if not os.path.exists(path2out):
    os.makedirs(path2out)

## load npy
path2npy    = path2root + "disparities{}.npy".format(surfix)

npy         = np.load(path2npy)


## get dimension
num, _, _   = npy.shape
#num    = 100
for n in np.arange(num):
    fName   = str(n).zfill(6)
    npy_    = npy[n,:,:]
    npy_    = np.array(Image.fromarray(npy_).resize([512,256], Image.BILINEAR))#LANCZOS))  
    disp_resized_np = npy_

    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    name_dest_im = os.path.join(path2out, "{}_disp.jpeg".format(fName))
    im.save(name_dest_im)

    #plt.imsave(os.path.join(path2out, "{}_disp.png".format(fName)), npy_, cmap='plasma')
    if n % 10 == 0:
        print("Status: {}%".format(np.round(n/num*100,2)))

print("Done!")



