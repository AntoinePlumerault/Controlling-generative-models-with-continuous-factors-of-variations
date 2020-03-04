# code largely inspired of https://github.com/tensorflow/lattice/tree/master/tensorflow_lattice/python/lib

import tensorflow as tf
import numpy as np
import tensorflow.keras as K



class Smoothness(K.regularizers.Regularizer):
    def __init__(self, l):
        self.l = l
    def __call__(self, x):
        x = tf.reshape(x, [1,-1,1])
        filters = tf.constant([[[-1.0]],[[1.0]]])
        regularizer = tf.nn.conv1d(x, 
            filters=filters, stride=1, padding='VALID')
        regularizer = self.l * tf.reduce_mean(tf.square(regularizer))
        return regularizer


class Lattice(K.layers.Layer):
    def __init__(self, **kwargs):
        self.input_min = kwargs.pop('input_min', -5)
        self.input_max = kwargs.pop('input_max', 5)
        self.lattice_size = kwargs.pop('lattice_size', 101)
        self.regularizer_amount = kwargs.pop('regularizer_amount', 1)
        super(Lattice, self).__init__(**kwargs)

        
        self.lattice_points = tf.reshape(
            np.linspace(
                self.input_min, 
                self.input_max, 
                self.lattice_size).astype(np.float32),
            [1, -1])
        print(self.lattice_points)
        self.range = self.input_max - self.input_min
    
    def build(self, input_shape):
        super(Lattice, self).build(input_shape)
        self.lattice_params = self.add_weight('lattice_params',
            shape=[1, self.lattice_size],
            initializer=tf.initializers.ones(),
            constraint=K.constraints.NonNeg(),
            regularizer=Smoothness(self.regularizer_amount),
            trainable=True)

    def call(self, inputs, training=False):
        net = tf.tile(inputs,[1,self.lattice_size]) - self.lattice_points  
        net = tf.maximum(
            tf.minimum(net, self.range/(self.lattice_size-1)/2.0),
            -self.range/(self.lattice_size-1)/2.0)
        net = tf.matmul(net, self.lattice_params, transpose_b=True)
        print(net, 'sdfyuhghjsgfjhsg')
        return net

    def compute_output_shape(self, input_shape):
        return input_shape