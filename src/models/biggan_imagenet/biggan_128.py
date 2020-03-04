from __future__ import absolute_import, division, print_function

import sys
sys.path.append('./')
#sys.path.append('../')

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from . import ops as O



meta = {
    'z_dim': 120,
    'img_size': [128, 128, 3],
    'n_cat': 1000,
}

def BigGan(sn=False):

    def Block(x, z, c, filters, name):
        z_c = K.layers.Concatenate(
            name=name+'/concat')([z, c])
        
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_1')([x, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_1')(net)
        net = K.layers.UpSampling2D(
            name=name+'/upsamle')(net)
        net = O.layers.Conv2D(filters, 3, padding='SAME', sn=sn,
            name=name+'/conv_1')(net)
        net = O.layers.AdaIN(sn=sn,
            name=name+'/adain_2')([net, z_c])
        net = K.layers.ReLU(
            name=name+'/relu_2')(net)
        net = O.layers.Conv2D(filters, 3, padding='SAME', sn=sn,
            name=name+'/conv_2')(net)
        
        res = K.layers.UpSampling2D(
            name=name+'/upsample_res')(x)
        res = O.layers.Conv2D(filters, 1, padding='SAME', sn=sn,
            name=name+'/conv_res')(res)

        return K.layers.Add(name=name+'/add')([res, net])


    z = K.Input(shape=[120], dtype=tf.float32,
        name='z') 
    c = K.Input(shape=[1000], dtype=tf.float32,
        name='c')

    embeded_c = K.layers.Dense(128, use_bias=False,
        name='embedding')(c)
    
    z_splits = K.layers.Lambda(lambda z: tf.split(z, 6, axis=1),
        name='split')(z)

    net = O.layers.Dense(4*4*16*96, sn=sn,
        name='dense')(z_splits[0])
    net = K.layers.Reshape([4,4,16*96],
        name='reshape')(net)
    net = Block(net, z_splits[1], embeded_c, 96*16, 
        name='block_1') 
    net = Block(net, z_splits[2], embeded_c, 96*8, 
        name='block_2') 
    net = Block(net, z_splits[3], embeded_c, 96*4, 
        name='block_3') 
    net = Block(net, z_splits[4], embeded_c, 96*2, 
        name='block_4') 
    net = O.layers.SelfAttention(filters=96*2, reduction=8, sn=sn, 
        name='attention')(net)   
    net = Block(net, z_splits[5], embeded_c, 96*1, 
        name='block_5') 
    net = K.layers.BatchNormalization(epsilon=1e-4,
        name='batch_normalization')(net)
    net = K.layers.ReLU(
        name='relu')(net)
    net = O.layers.Conv2D(3, 3, padding='SAME', sn=sn, 
        activation=tf.math.tanh, name='conv')(net)

    return K.Model(inputs=[z, c], outputs=net)
