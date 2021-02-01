## convert npy to image
'''
date:   2018-11-13
author: kuo
'''
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from shutil import copyfile
import os
from PIL import Image
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

save_source = not True
postproc = True
plasma = not True
if postproc:
    surfix = '_demo_ppp'
else:
    surfix = ''


## setup root path
slipt_type = 'kitti'

#model_name = 'c_kitti_resASPPcs_8x20_regh'
#model_name = 'monodepth2/eigen/'
#model_name = 'adareg/eigen/'
model_name = 'c_kitti_vggASPP_8x90/disp_cityscapes'
model_name = 'c_kitti_vggASPP_8x90/disp_deepdrive'
model_name = 'monodepth_eigen'
model_name = 'kitti_vggASPP_8x20_pr'
model_name = 'c_kitti_vggASPP_8x90'
#model_name = 'model_kitti'
path2root   = '../../models/{}/{}/'.format(model_name, 'disp_demo') #20_regh/'
#path2dataset= '../../data/dataset/Deepdrive/'
#path2dataset= '../../data/dataset/cityscapes/'

out_dir = '{}{}/'.format(slipt_type, surfix)
path2out = path2root + out_dir 
if not os.path.exists(path2out):
    os.makedirs(path2out)


if save_source:    
    path2npysrc    = path2root + "src.npy"
    path2src = path2root +'/src/'
    if not os.path.exists(path2src):
        os.makedirs(path2src)


if not os.path.exists(path2out):
    os.makedirs(path2out)

## load npy
path2npy    = path2root + 'stereo_640x192_eigen.npy' #"kitti_disparities{}.npy".format(surfix)
path2npy    = path2root + "disparities{}.npy".format(surfix)

npy         = np.load(path2npy)
if save_source:   
	npysrc         = np.load(path2npysrc)

## get dimension
num, _, _   = npy.shape
#num    = 100
for n in np.arange(num):
    fName   = str(n).zfill(6)
    npy_    = npy[n,:,:]
    npy_    = np.array(Image.fromarray(npy_).resize([1242,375], Image.BILINEAR))#LANCZOS))  
    #npy_    = np.array(Image.fromarray(npy_).resize([768,384], Image.BILINEAR))#LANCZOS))  
    if plasma:
        plt.imsave(os.path.join(path2out, "{}_disp.png".format(fName)), npy_, cmap='plasma')
    else:
        disp_resized_np = npy_
        vmax = np.percentile(disp_resized_np, 99)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(path2out, "{}_disp.png".format(fName))
        im.save(name_dest_im)

    if save_source:
        npysrc_ = npysrc[n,:,:]
        #npysrc_    = np.array(Image.fromarray(npysrc_).resize([768,384], Image.BILINEAR))#LANCZOS))  
        plt.imsave(os.path.join(path2src, "{}_src.jpeg".format(fName)), npysrc_) 

    if n % 10 == 0:
        print("Status: {}%".format(np.round(n/num*100,2)))
    if n > 1000:
        break

print("Done!")



