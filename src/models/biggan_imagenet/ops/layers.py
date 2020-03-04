from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.keras as K 



def _spectral_normalization(layer, kernel, name_depth=1):
    kernel_shape = kernel.shape.as_list()

    kernel_name = '/'.join(kernel.name[:-2].split('/')[-name_depth:])
    u = layer.add_weight('{}_u'.format(kernel_name),
        shape=[1, kernel_shape[-1]],
        initializer=tf.random_normal_initializer(),
        trainable=False)
    
    W = tf.reshape(kernel, [-1, kernel_shape[-1]])
    
    v_hat = tf.math.l2_normalize(
        tf.matmul(u, W, transpose_b=True), 
        epsilon=1e-4)
    u_hat = tf.math.l2_normalize(
        tf.matmul(v_hat, W), 
        epsilon=1e-4)

    sigma = tf.matmul(
        tf.matmul(v_hat, W), u_hat, transpose_b=True)

    update_u = u.assign(u_hat, 
        name='{}_u_update_op'.format(kernel_name))
    update_kernel = kernel.assign(kernel / sigma, 
        name='{}_update_op'.format(kernel_name))
            
    layer.add_update([update_u, update_kernel])

def _flatten_spatial_dims(inputs):
    shape = inputs.shape.as_list()
    return tf.reshape(inputs,[-1, shape[1]*shape[2], shape[-1]])

class Dense(K.layers.Dense):
    def __init__(self, *args, **kwargs):
        self.sn = kwargs.pop('sn')
        super(Dense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(Dense, self).build(input_shape)
        if self.sn: _spectral_normalization(self, self.kernel)

class Conv2D(K.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        self.sn = kwargs.pop('sn')
        super(Conv2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)
        if self.sn: _spectral_normalization(self, self.kernel)

class AdaIN(K.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.sn = kwargs.pop('sn')
        super(AdaIN, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(AdaIN, self).build(input_shape)
        
        x_shape, z_shape = input_shape
        # if tf.__version__[0] == '1':
        x_shape = x_shape.as_list() #[d.value for d in x_shape.as_list()]
        z_shape = z_shape.as_list() #[d.value for d in z_shape.as_list()]

        self.gamma_kernel = self.add_weight('gamma_kernel',
            shape=[z_shape[-1], x_shape[-1]],
            initializer=tf.initializers.orthogonal(),
            trainable=True)
        self.beta_kernel = self.add_weight('beta_kernel',
            shape=[z_shape[-1], x_shape[-1]], 
            initializer=tf.initializers.orthogonal(),
            trainable=True)

        self.moving_mean = self.add_weight('moving_mean', 
            shape=[x_shape[-1]], 
            initializer=tf.zeros_initializer(),
            trainable=False)
        self.moving_variance = self.add_weight('moving_variance',
            shape=[x_shape[-1]],
            initializer=tf.ones_initializer(),
            trainable=False)

        if self.sn:
            _spectral_normalization(self, self.gamma_kernel)
            _spectral_normalization(self, self.beta_kernel)


    def call(self, inputs, training=False):
        x, z = inputs
        ch = x.shape.as_list()[-1]

        mean, variance = self.moving_mean, self.moving_variance
        # mean, variance = tf.cond(
        #     training,
        #     lambda: tf.nn.moments(x, axes=[1, 2], keepdims=True),
        #     lambda: self.moving_mean, self.moving_variance)
        
        self.add_update([
            self.moving_mean.assign(
                0.999 * self.moving_mean + 0.001 * mean,
                name='update_moving_mean'), 
            self.moving_variance.assign(
                0.999 * self.moving_variance + 0.001 * variance,
                name='update_moving_variance')
        ], inputs=True)

        offset = tf.matmul(z, self.beta_kernel)
        scale = tf.matmul(z, self.gamma_kernel) + 1.0
        offset = tf.reshape(offset, [-1, 1, 1, ch])
        scale = tf.reshape(scale, [-1, 1, 1, ch])
        
        net = tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-4)

        return net

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttention(K.layers.Layer):

    def __init__(self, **kwargs):
        self.filters = kwargs.pop('filters')
        self.reduction = kwargs.pop('reduction')
        self.sn = kwargs.pop('sn')
        super(SelfAttention, self).__init__(**kwargs)
        
        if tf.__version__[0] != '1':
            self.f = Conv2D(
                filters=self.filters // self.reduction, use_bias=False,
                kernel_size=1, padding='SAME', sn=self.sn, name='f')
            self.g = Conv2D(
                filters=self.filters // self.reduction, use_bias=False,
                kernel_size=1, padding='SAME', sn=self.sn, name='g')
            self.h = Conv2D(
                filters=self.filters // 2, use_bias=False,
                kernel_size=1, padding='SAME', sn=self.sn, name='h')
            self.o = Conv2D(
                filters=self.filters, use_bias=False,
                kernel_size=1, padding='SAME', sn=self.sn, name='o')
    
    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        
        input_shape = input_shape.as_list()

        self.gamma = self.add_weight('gamma',
            shape=[1],
            initializer=tf.ones_initializer(),
            trainable=True)

        if tf.__version__[0] == '1':
            self.f_kernel = self.add_weight('f/kernel',
                shape=[1,1,input_shape[-1], input_shape[-1] // self.reduction],
                initializer=tf.initializers.orthogonal(),
                trainable=True)
            self.g_kernel = self.add_weight('g/kernel',
                shape=[1,1,input_shape[-1], input_shape[-1] // self.reduction],
                initializer=tf.initializers.orthogonal(),
                trainable=True)
            self.h_kernel = self.add_weight('h/kernel',
                shape=[1,1,input_shape[-1], input_shape[-1] // 2],
                initializer=tf.initializers.orthogonal(),
                trainable=True)
            self.o_kernel = self.add_weight('o/kernel',
                shape=[1,1,input_shape[-1] // 2, input_shape[-1]],
                initializer=tf.initializers.orthogonal(),
                trainable=True)
            
            if self.sn:
                _spectral_normalization(self, self.f_kernel, name_depth=2)
                _spectral_normalization(self, self.g_kernel, name_depth=2)
                _spectral_normalization(self, self.h_kernel, name_depth=2)
                _spectral_normalization(self, self.o_kernel, name_depth=2)
        else:
            with tf.name_scope('f'):
                self.f.build(input_shape)
            with tf.name_scope('g'):
                self.g.build(input_shape)
            with tf.name_scope('h'):
                self.h.build(input_shape)
            with tf.name_scope('o'):
                self.o.build(input_shape[:-1]+[input_shape[-1]//2])


    def call(self, inputs, training=False):
        shape = inputs.shape.as_list()

        if tf.__version__[0] == '1':
            f = tf.nn.conv2d(inputs, 
                filter=self.f_kernel, strides=[1,1,1,1], padding='SAME')
            g = tf.nn.conv2d(inputs, 
                filter=self.g_kernel, strides=[1,1,1,1], padding='SAME')
        else:
            f = self.f(inputs) 
            g = self.g(inputs)
        
        g = tf.nn.max_pool(g, ksize=[1,2,2,1], strides=[1,2,2,1], 
            padding='SAME')
    
        beta = tf.math.softmax(tf.linalg.matmul(
            _flatten_spatial_dims(f),
            _flatten_spatial_dims(g),
            transpose_b=True))

        if tf.__version__[0] == '1':
            h = tf.nn.conv2d(inputs,
                filter=self.h_kernel, strides=[1,1,1,1], padding='SAME')
        else:
            h = self.h(inputs)
        
        h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], 
            padding='SAME')

        net = tf.matmul(beta, _flatten_spatial_dims(h))
        net = tf.reshape(net, [-1, shape[1], shape[2], shape[3] // 2])        
        
        if tf.__version__[0] == '1':
            net = tf.nn.conv2d(net,
                filter=self.o_kernel, strides=[1,1,1,1], padding='SAME')
        else:
            net = self.o(net)

        return inputs + self.gamma * net

    def compute_output_shape(self, input_shape):
        return input_shape
