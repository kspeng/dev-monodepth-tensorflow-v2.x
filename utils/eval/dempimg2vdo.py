import cv2
import numpy as np
import glob
import time
import imageio
import os

model_name = 'model_kitti'
model_name_2 = 'c_kitti_vggASPP_8x90'
path2root   = '../../models/{}/{}/'.format(model_name, 'disp_demo_plasma') #20_regh/'
path2root   = '../../models/{}/{}/'.format(model_name, 'disp_demo') #20_regh/'
path2root   = '../../data/dataset/kitti/data/2011_10_03/2011_10_03_drive_0034_sync/image_02/data/'
path2root_2 = '../../models/{}/{}/'.format(model_name_2, 'disp_demo') #20_regh/'
#path2img = 'C:/New folder/Images/'
stt = time.time()
img_array_1 = []
f_thr_i, f_thr_e = 680, 940
#for filename in sorted(glob.glob(path2root+'kitti_demo_pp/*.*')):
for i,filename in enumerate(sorted(glob.glob(path2root+'*.*'))):
    if i > f_thr_i and i < f_thr_e:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640,192), interpolation = cv2.INTER_AREA)
        img_array_1.append(img)
    #break
print('>>> load reference. ')

img_array = []
for i, filename in enumerate(sorted(glob.glob(path2root_2+'kitti_demo_ppp/*.*'))):
    if i > f_thr_i and i < f_thr_e:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640,192), interpolation = cv2.INTER_AREA)
        img = np.vstack((img_array_1[i-f_thr_i-1], img))
        img_array.append(img)
    #break 
print('>>> load results and combine. ')

#img_array = []
#for i in range(len(img_array_1)):

print(img_array[0].shape)
'''
img_array = img_array_1
'''
path2root   = '../../models/{}/{}/'.format(model_name_2, 'disp_demo') #20_regh/'
#imageio.mimsave(path2root+'demo-iros.gif', img_array, duration = 0.1)

height, width, layers = img_array[0].shape
size = (width,height)    
out = cv2.VideoWriter(path2root+'demo-iros.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(img_array)):
    out.write(img_array[i])
print('exe time: ', time.time()-stt) 
out.release()
