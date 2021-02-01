'''
MODEL    :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-24
FILE     :: utils_nn.py 
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class NnKits(object):
    """utils for neural network"""

    def __init__(self):
        pass

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.compat.v1.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def downsample_nn(self, x, ratio, method='NN'):
        s = tf.shape(x)
        h = tf.cast(s[1] / ratio, tf.int32)
        w = tf.cast(s[2] / ratio, tf.int32)
        if method == 'bilinear':
            return tf.image.resize_bilinear(x, [h, w], align_corners=True)
        else:
            return tf.compat.v1.image.resize_nearest_neighbor(x, [h, w], align_corners=True)

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def upconv(self, x, num_layers, kernel_size, scale, type='conv'):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_layers, kernel_size, 1)
        return conv

    def conv(self, x, num_layers, kernel_size, stride, normalizer_fn=None, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_layers, kernel_size, stride, 'VALID', normalizer_fn=normalizer_fn, activation_fn=activation_fn)

    def conv_block(self, x, num_layers, kernel_size, stride=2):
        conv1 = self.conv(x,     num_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_layers, kernel_size, stride)
        return conv2

    # Resdual module

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_out_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_out_layers, 1)
        out = self.resconv(out, num_out_layers, 2)
        return out

    def res33module(self, x, num_layers, stride=1, num_blocks=2, activation_fn='relu'):
        activation_fn = self.activate_sel(activation_fn) 
        bn=slim.batch_norm # None 
      
        in_channel = x.get_shape().as_list()[-1]
        if in_channel == num_layers:
            if stride == 1:
                shortcut = x
            else:
                shortcut = self.maxpool(x,3)                
        else:
            shortcut = self.conv(x, num_layers, 1, stride)

        # Residual
        if num_blocks==2:
            x = self.conv(x, num_layers, 3, stride,
                normalizer_fn = bn, activation_fn=activation_fn)
            x = self.conv(x, num_layers, 3, 1,
                normalizer_fn = bn, activation_fn=None)
        else:
            x = self.conv(x, num_layers, 3, stride,
                normalizer_fn = bn, activation_fn=None)

        x = x + shortcut
        x = activation_fn(x)

        return x 

    def res33module_be(self, x, num_layers, stride=1, activation_fn='elu'): # pe_new 20597408
        activation_fn = self.activate_sel(activation_fn) 
        bn=slim.batch_norm # None

        shortcut = self.conv(x, num_layers, 3, stride, 
            normalizer_fn = bn, activation_fn=activation_fn)    

        # Residual
        x = self.conv(x, num_layers, 3, stride,
            normalizer_fn = bn, activation_fn=activation_fn)
        x = self.conv(x, num_layers, 3, 1,
            normalizer_fn = bn, activation_fn=None)
    
        x = x + shortcut
        x = activation_fn(x)

        return x  

    def res33module_bes(self, x, num_layers, stride=1, activation_fn='relu'):        
        activation_fn = self.activate_sel(activation_fn) 
        bn=slim.batch_norm # None

        x = self.conv(x, num_layers, 3, stride,
            normalizer_fn = bn, activation_fn=activation_fn)
        
        shortcut = x
        
        x = self.conv(x, num_layers, 3, 1,
            normalizer_fn = bn, activation_fn=None)

        x = x + shortcut
        x = activation_fn(x)

        return x 

    def res33module_besd(self, x, num_layers, stride=1, activation_fn='relu'):        
        activation_fn = self.activate_sel(activation_fn) 
        bn=slim.batch_norm # None

        x = self.conv(x, num_layers, 3, 1,
            normalizer_fn = bn, activation_fn=activation_fn)

        x = self.conv(x, num_layers, 3, stride,
            normalizer_fn = bn, activation_fn=activation_fn)

        shortcut = x
        
        x = self.conv(x, num_layers, 3, 1,
            normalizer_fn = bn, activation_fn=None)
        x = self.conv(x, num_layers, 3, 1,
            normalizer_fn = bn, activation_fn=None)        

        x = x + shortcut
        x = activation_fn(x)

        return x 

    def res33block(self, x, num_layers, stride=2, num_blocks=2, method='baseline'):#, activation_fn='relu', stride_back=False):
        # parameters
        activation_fn = 'elu' 
        stride_back = True
        module_stride_back = True        

        # module seeting
        if method == 'be':
            convbolck = self.res33module_be  
        elif method == 'bes':
            convbolck = self.res33module_bes
        elif method == 'besd':
            convbolck = self.res33module_besd            
        elif method == 'baseline':
            convbolck = self.res33module
            activation_fn = 'relu' 
            stride_back = False    
        else: 
            print('>> Unsupport encoder type!!')
            exit()
        
        # block flow   
        if method=='baseline':
            x = convbolck(x, num_layers, stride, activation_fn=activation_fn)    
            x = convbolck(x, num_layers, 1, activation_fn=activation_fn)  
        else:
            x = convbolck(x, num_layers, 1, activation_fn=activation_fn)                    
            x = convbolck(x, num_layers, stride, activation_fn=activation_fn)    

        return x


    def deconv(self, x, num_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    # Atrous helper
    def atrous_conv(self, x, num_layers, kernel_size, rate, apply_bn_first=True):
        pk = np.floor((kernel_size - 1) / 2).astype(np.int32)
        pr = rate - 1
        p = pk + pr
        out = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])

        if apply_bn_first is True:
            out = slim.batch_norm(out)

        out = tf.nn.relu(out)
        out = slim.conv2d(out, num_layers * 2, 1, 1, 'VALID')
        out = slim.batch_norm(out)
        out = tf.nn.relu(out)
        out = slim.conv2d(out, num_layers, kernel_size=kernel_size, stride=1, rate=rate, padding='VALID',
                          activation_fn=None, normalizer_fn=None)

        return out

    def activate_sel(self, type='relu'):

        if type == 'elu':
            fn = tf.nn.elu
        elif type == 'relu':
            fn = tf.nn.relu
        else:
            print('>> Unsupport activation type!!')
            exit()   
                    
        return fn

    def dec_iconv_block(self, enc, skip=None, udisp=None, num_layers=256, kernel_size=3):
        # set convenience functions  
        conv = self.conv
        upconv = self.upsample_conv
        upconv_ = upconv(enc,   num_layers, kernel_size, 2) #H/32

        if udisp is None:
            concat_ = tf.concat([upconv_, skip], 3)
        elif skip is None:
            concat_ = tf.concat([upconv_, udisp], 3)   
        else:
            concat_ = tf.concat([upconv_, skip, udisp], 3)        
        iconv_  = conv(concat_,   num_layers, kernel_size, 1)
        return iconv_
