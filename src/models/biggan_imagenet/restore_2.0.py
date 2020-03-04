from __future__ import absolute_import, division, print_function

import sys
sys.path.append('../..')

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as K
import matplotlib.pyplot as plt

from absl import app, flags, logging

import biggan_128, biggan_256, biggan_512, biggan_deep_128, biggan_deep_256, biggan_deep_512

import biggan_deep_512
MODELS = {
    'biggan-128': biggan_128,
    'biggan-256': biggan_256,
    'biggan-512': biggan_512,
    'biggan-deep-128': biggan_deep_128,
    'biggan-deep-256': biggan_deep_256,
    'biggan-deep-512': biggan_deep_512,
}

def main(argv):

    # Create model
    model_name_template = 'biggan{}-{}'
    model_name = model_name_template.format(
        '-deep' if FLAGS.deep else '',
        FLAGS.image_size
    )

    model = MODELS[model_name].BigGan(sn=True)
    model.build(input_shape=[(1, 120), (1,1000)])
    
    for variable in model.variables:
        print(variable.name, ' --- ', variable.shape)
        if variable.name == 'attention/f/kernel:0':
            print(variable)
    
    model.load_weights(os.path.join(
        'weights', 'biggan{}_{}.h5'.format(
            '-deep' if FLAGS.deep else '',
            FLAGS.image_size)
    ), by_name=True)
    
    z_dim = MODELS[model_name].meta['z_dim']
    z = np.float32(np.random.randn(1, z_dim))
    c = np.float32(np.eye(1000)[[14]])
    
    image = model([z, c], training=False)
    plt.imshow((image[0]+1)/2)
    plt.show()

    
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_boolean('deep', False, '')
    flags.DEFINE_enum('image_size', '128', ['128', '256', '512'], '')
    app.run(main)
