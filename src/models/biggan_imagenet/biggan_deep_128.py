from __future__ import absolute_import, division, print_function

import sys
sys.path.append('./')

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from . import ops as O


meta = {
    'z_dim': 128,
    'img_size': [128, 128, 3],
    'n_cat': 1000,
}

def BigGan(sn=False):

    def Block(x, z_c, inputs_ch, outputs_ch, name, up=False):
        
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_1')([x, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_1')(net)
        net = O.layers.Conv2D(inputs_ch//4, 1, padding='SAME', sn=sn,
            name=name+'/conv_1')(net)
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_2')([net, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_2')(net)
        if up:
            net = K.layers.UpSampling2D(
                name=name+'/upsample')(net)
        net = O.layers.Conv2D(inputs_ch//4, 3, padding='SAME', sn=sn,
            name=name+'/conv_2')(net)
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_3')([net, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_3')(net)
        net = O.layers.Conv2D(inputs_ch//4, 3, padding='SAME', sn=sn,
            name=name+'/conv_3')(net)
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_4')([net, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_4')(net)
        net = O.layers.Conv2D(outputs_ch, 1, padding='SAME', sn=sn,
            name=name+'/conv_4')(net)
        
        res = K.layers.Lambda(lambda x: x[:,:,:,:outputs_ch],
            name=name+'/drop_channel')(x)
        if up:
            res = K.layers.UpSampling2D(
                name=name+'/upsample_res')(res)
        

        return K.layers.Add(name=name+'/add')([res, net])


    z = K.Input(shape=[128], dtype=tf.float32,
        name='z') 
    c = K.Input(shape=[1000], dtype=tf.float32,
        name='c')

    embeded_c = K.layers.Dense(128, use_bias=False,
        name='embedding')(c)
    
    z_c = K.layers.Concatenate(
            name='concat')([z, embeded_c])
    ch = 128
    net = O.layers.Dense(4*4*16*ch, sn=sn,
        name='dense')(z_c)
    net = K.layers.Reshape([4,4,16*ch],
        name='reshape')(net)
    net = Block(net, z_c, ch*16, ch*16, 
        name='block_1') 
    net = Block(net, z_c, ch*16, ch*16, up=True, 
        name='block_2') 
    net = Block(net, z_c, ch*16, ch*16, 
        name='block_3') 
    net = Block(net, z_c, ch*16, ch*8, up=True,
        name='block_4') 
    net = Block(net, z_c, ch*8, ch*8, 
        name='block_5') 
    net = Block(net, z_c, ch*8, ch*4, up=True,
        name='block_6') 
    net = Block(net, z_c, ch*4, ch*4, 
        name='block_7') 
    net = Block(net, z_c, ch*4, ch*2, up=True,
        name='block_8') 
    net = O.layers.SelfAttention(filters=ch*2, reduction=8, sn=sn, 
        name='attention')(net)   
    net = Block(net, z_c, ch*2, ch*2, 
        name='block_9') 
    net = Block(net, z_c, ch*2, ch*1, up=True, 
        name='block_10') 
    net = K.layers.BatchNormalization(epsilon=1e-4,
        name='batch_normalization')(net)
    net = K.layers.ReLU(
        name='relu')(net)
    net = O.layers.Conv2D(128, 3, padding='SAME', sn=sn, 
        activation=tf.math.tanh, name='conv')(net)
    net = K.layers.Lambda(lambda x: x[:,:,:,:3])(net)
    
    return K.Model(inputs=[z, c], outputs=net)
