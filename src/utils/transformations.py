import tensorflow as tf



def translate_horizontally(images, dx):
    img_shape = images.shape.as_list()
    batch_size, size = img_shape[0], img_shape[-2]
    delta = (int)(dx * (float)(size))
    zeros = tf.zeros([batch_size, size, abs(delta), 3])
    if delta > 0:
        new_images = tf.concat(
            [zeros, images[:,:, :size-delta, :]], 
            axis=-2) 
        masks = tf.concat(
            [zeros, tf.ones([batch_size, size, size-abs(delta), 3])], 
            axis=-2) 
    else:
        new_images = tf.concat(
            [images[:,:, abs(delta):, :], zeros], 
            axis=-2) 
        masks = tf.concat(
            [tf.ones([batch_size, size, size-abs(delta), 3]), zeros], 
            axis=-2) 
    
    return new_images, masks


def translate_vertically(images, dy):
    img_shape = images.shape.as_list()
    batch_size, size = img_shape[0], img_shape[-2]
    num_channels = 3
    delta = (int)(dy * (float)(size))
    zeros = tf.zeros([batch_size, abs(delta), size, 3])
    if delta > 0:
        new_images = tf.concat(
            [zeros, images[:, :size-delta, :, :]], 
            axis=-3) 
        masks = tf.concat(
            [zeros, tf.ones([batch_size, size-abs(delta), size, 3])], 
            axis=-3) 
    else:
        new_images = tf.concat(
            [images[:, abs(delta):, :, :], zeros], 
            axis=-3) 
        masks = tf.concat(
            [tf.ones([batch_size, size-abs(delta), size, 3]), zeros], 
            axis=-3) 
    return new_images, masks


def zoom(images, s):
    s = 1.0 + s
    size = images.shape.as_list()[-2]
    num_channels = 3
    new_size = int(float(size) / s)
    new_images = tf.image.resize_with_crop_or_pad(images, new_size, new_size)
    new_images = tf.image.resize(new_images, [size, size])

    masks = tf.ones_like(images)
    masks = tf.image.resize_with_crop_or_pad(masks, new_size, new_size)
    masks = tf.image.resize(masks, [size, size])
    return new_images, masks


def brightness(images, b):
    size = images.shape.as_list()[-2]
    num_channels = 3
    new_images = tf.clip_by_value(images + b, -1, 1)
    masks = tf.ones_like(images)

    return new_images, masks