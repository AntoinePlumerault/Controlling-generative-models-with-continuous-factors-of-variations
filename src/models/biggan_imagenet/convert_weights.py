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

MODELS = {
    'biggan-128': biggan_128,
    'biggan-256': biggan_256,
    'biggan-512': biggan_512,
    'biggan-deep-128': biggan_deep_128,
    'biggan-deep-256': biggan_deep_256,
    'biggan-deep-512': biggan_deep_512,
}

def main(argv):
    # Load module
    url_template = 'https://tfhub.dev/deepmind/biggan{}-{}/{}'
    url = url_template.format(
        '-deep' if FLAGS.deep else '',
        FLAGS.image_size,
        '1' if FLAGS.deep else '2')
    
    module = hub.Module(url)
    print('\n'.join(v.name+' '+str(v.shape) for v in module.variables))
    print('----------------------------------------------')

    # Create model
    model_name_template = 'biggan{}-{}'
    model_name = model_name_template.format(
        '-deep' if FLAGS.deep else '',
        FLAGS.image_size)

    model = MODELS[model_name].BigGan(sn=True)
    model.build(input_shape=[(1, 120), (1,1000)])

    module_var_names = [v.name for v in module.variables]
    model_var_names = [v.name for v in model.variables]
    model_var_dict = {v.name: v for v in model.variables}
    module_var_dict = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        module_var_values = sess.run(module.variables)
        module_var_dict = {name: value for name, value in zip(module_var_names, module_var_values)}
    
    model_to_module_dict = {}

    for name in model_var_names:
        
        if FLAGS.deep:
            ema = 'ema_0.9999'
        else:
            ema = 'ema_b999900'
        print(name, end='  ---  ')
        model_var_name = name

        if 'embedding' not in name:
            name = 'module/Generator/' + name
        else:
            name = 'module/' + name

        name = name.replace('block_1/', 'GBlock/')
        for i in range(1, 15):
            name = name.replace('block_{}/'.format(i+1), 'GBlock_{}/'.format(i))

            name = name.replace('embedding', 'linear')
        if FLAGS.deep:
            name = name.replace('/dense/', '/GenZ/G_linear/')
        else:
            name = name.replace('/dense/', '/G_Z/G_linear/')

        for i in range(1,5):
            if i == 1:
                I = ''
            else:
                I = '_{}'.format(i-1)
            
            if FLAGS.deep:
                name = name.replace('/adain_{}/gamma_kernel_u'.format(i), 
                    '/BatchNorm{}/scale/u0'.format(I))
                name = name.replace('/adain_{}/beta_kernel_u'.format(i), 
                    '/BatchNorm{}/offset/u0'.format(I))

                name = name.replace('/adain_{}/gamma_kernel'.format(i), 
                    '/BatchNorm{}/scale/w/{}'.format(I, ema))
                name = name.replace('/adain_{}/beta_kernel'.format(i), 
                    '/BatchNorm{}/offset/w/{}'.format(I, ema))
                
                name = name.replace('/adain_{}/moving_mean'.format(i), 
                    '/BatchNorm{}/accumulated_mean'.format(I))
                name = name.replace('/adain_{}/moving_variance'.format(i), 
                    '/BatchNorm{}/accumulated_var'.format(I))
            
            else:
                name = name.replace('/adain_{}/gamma_kernel_u'.format(i), 
                    '/HyperBN{}/gamma/u0'.format(I))
                name = name.replace('/adain_{}/beta_kernel_u'.format(i), 
                    '/HyperBN{}/beta/u0'.format(I))

                name = name.replace('/adain_{}/gamma_kernel'.format(i), 
                    '/HyperBN{}/gamma/w/{}'.format(I, ema))
                name = name.replace('/adain_{}/beta_kernel'.format(i), 
                    '/HyperBN{}/beta/w/{}'.format(I, ema))
                
                name = name.replace('/adain_{}/moving_mean'.format(i), 
                    '/CrossReplicaBN{}/accumulated_mean'.format(I))
                name = name.replace('/adain_{}/moving_variance'.format(i), 
                    '/CrossReplicaBN{}/accumulated_var'.format(I))

        name = name.replace('attention/g/', 'attention/phi/')
        name = name.replace('attention/h/', 'attention/g/')
        name = name.replace('attention/f/', 'attention/theta/')
        name = name.replace('attention/o/', 'attention/o_conv/')

        name = name.replace('attention/gamma', 'attention/gamma/{}'.format(ema))

        for i in range(1,5):
            name = name.replace('/conv_{}/'.format(i), '/conv{}/'.format(i-1))
        
        if FLAGS.deep:
            name = name.replace('/conv/', '/conv_to_rgb/')
        else:
            name = name.replace('/conv_res/', '/conv_sc/')
            name = name.replace('/conv/', '/conv_2d/')
       
        name = name.replace('/kernel_u', '/u0')


        name = name.replace('kernel', 'w/{}'.format(ema))
        name = name.replace('bias', 'b/{}'.format(ema))

        if FLAGS.deep:
            name = name.replace('/batch_normalization/gamma', 
                '/BatchNorm/scale/{}'.format(ema))
            name = name.replace('/batch_normalization/beta', 
                '/BatchNorm/offset/{}'.format(ema))

            name = name.replace('/batch_normalization/moving_mean', 
                '/BatchNorm/accumulated_mean')
            name = name.replace('/batch_normalization/moving_variance', 
                '/BatchNorm/accumulated_var')
        else:
            name = name.replace('/batch_normalization/gamma', 
                '/ScaledCrossReplicaBN/gamma/{}'.format(ema))
            name = name.replace('/batch_normalization/beta', 
                '/ScaledCrossReplicaBN/beta/{}'.format(ema))

            name = name.replace('/batch_normalization/moving_mean', 
                '/ScaledCrossReplicaBNbn/accumulated_mean')
            name = name.replace('/batch_normalization/moving_variance', 
                '/ScaledCrossReplicaBNbn/accumulated_var')

        model_to_module_dict[model_var_name] = name

        if name not in module_var_names:
            print('', end='  !!!  ')
        print(name)

    assign_ops = []
    for variable in model.variables:
        print(variable.name, ' --- ', variable.shape)
        assign_ops.append(variable.assign(tf.reshape(
            module_var_dict[model_to_module_dict[variable.name]],
            variable.shape)))

    update_ops = []
    for update in model.updates:      
        if 'update_op' in update.name:
            update_ops.append(update)
  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(assign_ops)
        sess.run(update_ops)
        model.save_weights(os.path.join(
            'biggan_imagenet', 'weights', 'biggan{}_{}.h5'.format(
                '-deep' if FLAGS.deep else '',
                FLAGS.image_size,
                '1' if FLAGS.deep else '2')
            ))
        model.load_weights(os.path.join(
            'biggan_imagenet', 'weights', 'biggan{}_{}.h5'.format(
                '-deep' if FLAGS.deep else '',
                FLAGS.image_size,
                '1' if FLAGS.deep else '2')
            ), by_name=True)
        
        z_dim = MODELS[model_name].meta['z_dim']
        z = np.random.randn(1, z_dim)
        c = np.eye(1000)[[14]]

        image = sess.run(model([tf.cast(z, tf.float32), tf.cast(c, tf.float32)]))
        print(image)
        plt.imshow((image[0]+1)/2)
        plt.show()

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_boolean('deep', False, '')
    flags.DEFINE_enum('image_size', '128', ['128', '256', '512'], '')
    app.run(main)