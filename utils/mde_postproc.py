'''
MODEL	:: Tensorflow Computer Vision Platform
DATE    :: 2020-01-23
FILE 	:: postproc.py 
'''
import numpy as np
import tensorflow as tf

class MdePostproc:

    def __init__(self, left):
        self.left = left
        self.post_proc_init()

    def post_proc_init(self, reserve_rate=0.05, reserve_rate_plus=0.02):
        h = self.left.get_shape().as_list()[1]
        w = self.left.get_shape().as_list()[2]
        l, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

        #self.l_mask = 1.0 - np.clip(20 * (l - (0.08*v+0.02)), 0, 1)
        self.l_mask = 1.0 - np.clip(20 * (l - reserve_rate), 0, 1)        
        self.r_mask = np.fliplr(self.l_mask)    
        self.m_mask = (1.0 - self.l_mask - self.r_mask)    

        self.l_mask_plus = 1.0 - np.clip(20 * (l - reserve_rate_plus), 0, 1)
        self.r_mask_plus = np.fliplr(self.l_mask_plus)    
        self.m_mask_plus = (1.0 - self.l_mask_plus - self.r_mask_plus)      

    # Average-Type Post-processing
    def post_process(self, disp):
        l_disp = disp[0,:,:,0]
        r_disp = tf.image.flip_left_right(disp[1,:,:,:])[:,:,0]        
        m_disp = (l_disp + r_disp) / 2
        return self.r_mask * l_disp + self.l_mask * r_disp + self.m_mask * m_disp

    # Advanced Post-processing
    def get_nedgex(self, disp, edge_gain=1.):
        edge = abs(tf.image.sobel_edges(disp)[:,:,:,0])
        edge_max, edge_min = tf.reduce_max(edge), tf.reduce_min(edge)        
        return tf.sigmoid(((edge - edge_min) / ( edge_max -  edge_min)-0.5)*edge_gain)          

    def post_process_adv(self, disp):
        l_disp = disp[0,:,:,0]
        r_disp = tf.image.flip_left_right(disp[1,:,:,:])[:,:,0]        
        m_disp = tf.minimum(l_disp, r_disp) 
        a_disp = (l_disp + r_disp) / 2

        edge = self.get_nedgex(tf.expand_dims(tf.expand_dims(m_disp,0),3))[0,:,:,0] 
        edge = edge * self.m_mask

        m_disp = (1-edge) * m_disp + edge * a_disp
        
        return self.r_mask * l_disp + self.l_mask * r_disp + self.m_mask * m_disp

    # Edge-Guided Post-processing
    def get_right_edge(self, disp, kernel_size=21):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        
        disp_p = tf.pad(disp, [[0, 0], [1, 1], [p, p], [0, 0]])
        disp_p_ap = tf.nn.avg_pool2d(disp_p, ksize=[1, 3, p, 1], strides=[1, 1, 1, 1], padding='SAME')

        gx = disp_p_ap[:,1:-1,:(-2*p),:] - disp_p_ap[:,1:-1,(p+1):(-p+1),:]

        gx_max, gx_min = tf.reduce_max(gx), tf.reduce_min(gx)

        return tf.sigmoid((((gx - gx_min) / ( gx_max -  gx_min))-0.5)*32)

    def post_process_plus(self, disp):    
        gx = self.get_right_edge(disp)

        l_disp = disp[0,:,:,0]
        r_disp = tf.image.flip_left_right(disp[1,:,:,:])[:,:,0]

        gx_l = gx[0,:,:,0]
        gx_r = tf.image.flip_left_right(gx[1,:,:,:])[:,:,0]
        self.gx_l = gx_l
        self.gx_r = gx_r
        
               
        gx_w = tf.add(gx_l, gx_r)   
        gx_l = tf.truediv(gx_l, gx_w)
        gx_r = tf.truediv(gx_r, gx_w)
        m_disp = l_disp * gx_l + r_disp * gx_r

        return self.r_mask_plus * l_disp + self.l_mask_plus * r_disp + self.m_mask_plus * m_disp


