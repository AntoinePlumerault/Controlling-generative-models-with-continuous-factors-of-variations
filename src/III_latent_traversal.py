#

import os
import numpy as np
import tensorflow as tf

from absl import app, flags, logging

from models.biggan_imagenet import *
from utils.save_images import save_images
# from utils.losses import blur_MSE

MODELS = {
    'biggan_128': biggan_128,
    'biggan_256': biggan_256,
    'biggan_512': biggan_512,
    'biggan-deep_128': biggan_deep_128,
    'biggan-deep_256': biggan_deep_256,
    'biggan-deep_512': biggan_deep_512,
}

z_dim = 120 # latent space dimension
n_cat = 1000 # number of categories

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_traversals', 1, 
    "number of latent traversals to generate")
flags.DEFINE_integer('batch_size_III', 10, "")

def latent_traversal():

    experiment_dir = os.path.join(
        FLAGS.output_dir, 
        '{}'.format(FLAGS.transform), 
        '{}'.format(FLAGS.y))

# =============================== LOAD MODEL ===================================
    
    model_name = 'biggan{}_{}'.format(
        '-deep' if FLAGS.deep else '', FLAGS.image_size)
    
    model = MODELS[model_name].BigGan(sn=True)
    model.build([(1, z_dim), (1, n_cat)])
    model.load_weights(os.path.join(
        'models', 'biggan_imagenet', 'weights', '{}.h5'.format(model_name)), 
        by_name=True)

# ======================== DEFINE COMPUTATION GRAPH ============================
    
    z = tf.Variable(
        initial_value=tf.zeros([FLAGS.batch_size_III, z_dim]))
    c = tf.Variable(
        initial_value=np.tile(np.eye(1000)[[FLAGS.y]], [FLAGS.batch_size_III, 1]))
    alpha = tf.Variable(
        initial_value=tf.zeros([]))

    @tf.function
    def project(u, v):
        dot = tf.linalg.tensordot(u, v, axes=[[-1],[-1]])
        dot = tf.reshape(dot, [-1, 1])
        u_proj_v = u - dot*tf.reshape(v, [1,-1])
        return u_proj_v
    
    @tf.function
    def generate_image():
        x = model([z, c], training=False)
        return x
    
# ========================= GENERATE LATENT TRAVERSAL ==========================
    
    os.makedirs(os.path.join(
        experiment_dir, 'latent_traversals'), exist_ok=True)

    # read the direction
    filename = os.path.join(experiment_dir, 'directions', 'direction.csv')
    with open(filename, 'r') as f:
        u = [float(e) for e in f.readline().split(', ')]
        u = tf.cast(u, tf.float32)

    for batch in range(FLAGS.num_traversals // FLAGS.batch_size_III):
        z_0 = tf.random.normal([FLAGS.batch_size_III, z_dim])
        for alpha in np.linspace(-5.0, 5.0, 11):
            z.assign(project(z_0, u) + alpha * u)
            x = generate_image()

            for i in range(FLAGS.batch_size_III):
                save_images(x.numpy()[i:i+1,:,:,:], os.path.join(
                    experiment_dir, 'latent_traversals',
                        '{}_alpha={:2.1f}.png'.format(
                        batch*FLAGS.batch_size_III + i, (alpha+5)/10)))


if __name__ == '__main__':
    # output directory
    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs', 'test'), 
        "output directory")

    # choice of the model
    flags.DEFINE_enum('image_size', '128', ['128', '256', '512'], 
        "generated image size")
    flags.DEFINE_boolean('deep', False, 
        "use `deep` version of biggan or not")
    
    # choice of category and transformation
    flags.DEFINE_integer('y', 1, 
        "the category of the generated images")
    flags.DEFINE_enum('transform', 'horizontal_position', 
        ['horizontal_position', 'vertical_position', 'scale', 'brightness'], 
        "choice of transformation")
    
    app.run(latent_traversal)
