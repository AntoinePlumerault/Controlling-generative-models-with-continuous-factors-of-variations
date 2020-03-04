import tensorflow as tf
from numpy import pi


def blurred_mse(image_1, image_2, mask=None, sigma=3.0):
    """Compute the pixelwise l2 loss between blurred images.
    Inputs:
        image_1 & image_2: two 4D tensors
        mask: a 4D tensor to mask the images
        sigma: a float. The standard deviation of the gaussian kernel"""
    if mask is None:
        mask = tf.ones_like(image_1)
    ch = 3#image_1.shape.as_list()[-1]
    
    def gaussian_kernel():
        S = int(round(6*sigma))
        # S = tf.cast(tf.round(6*sigma), tf.int32)
        x = tf.cast(tf.reshape(tf.linspace(-S/2, S/2, S), [S, 1]), tf.float32)
        y = tf.cast(tf.reshape(tf.linspace(-S/2, S/2, S), [1, S]), tf.float32)
        kernel = 1 / (tf.sqrt(2*pi)*sigma) * tf.exp(-(x**2 + y**2)/(2*sigma**2))
        kernel = tf.reshape(kernel, [S,S,1])
        kernel = tf.stack([kernel]*ch, axis=2)
        return kernel
    
    error = image_1 - image_2
    error = tf.nn.depthwise_conv2d(error, gaussian_kernel(), [1,1,1,1], 'SAME' )
    error = tf.reduce_mean(tf.square(error) * mask, axis=[1,2,3]) / tf.reduce_mean(mask)
    return error