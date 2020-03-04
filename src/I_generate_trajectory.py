# Create a tf-record containing trajectories in the latent space which 
# corresponds to translations along the x axis. The scripts also outputs the 
# images.

import os
import numpy as np
import tensorflow as tf

from absl import app, flags, logging

from models.biggan_imagenet import *

from utils import transformations
from utils import reconstruction_errors
from utils.save_images import save_images

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

# save images ?
flags.DEFINE_bool('save_images', False, 
    "save images and tfrecord or just tfrecords")

flags.DEFINE_integer('num_trajectories', 1, 
    "number of trajectories to generate")
flags.DEFINE_integer('batch_size', 1, 
    "size of batch")
flags.DEFINE_integer('n_steps', 1, 
    "number of optimization steps for each intermediate target")
flags.DEFINE_boolean('renorm', True, 
    "Use normalization of the latent code")
flags.DEFINE_float('sigma', 3.0, 
    "sigma for the blur l2 loss")

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_trajectory():
    
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

    if FLAGS.transform == 'horizontal_position':
        transform = transformations.translate_horizontally
    if FLAGS.transform == 'vertical_position':
        transform = transformations.translate_vertically
    if FLAGS.transform == 'scale':
        transform = transformations.zoom
    if FLAGS.transform == 'brightness':
        transform = transformations.change_brightness

# ======================== DEFINE COMPUTATION GRAPH ============================
    
    z = tf.Variable( # latent code
        initial_value=tf.zeros([FLAGS.batch_size, z_dim]))
    c = tf.Variable( # one hot vector for category
        initial_value=np.tile(np.eye(1000)[[FLAGS.y]], [FLAGS.batch_size, 1]))
    T = tf.Variable( # transformation
        initial_value=tf.zeros([]))
    optimizer = tf.optimizers.Adam(0.01)

    def renormalize(z):
        z.assign(tf.clip_by_norm(z, tf.math.sqrt(float(z_dim)), axes=[-1]))
    
    @tf.function
    def generate_image(z, c):
        x = model([z, c], training=False)
        return x
    
    @tf.function
    def initialize_optimizer():
        for variable in optimizer.variables():
            variable.assign(variable.initial_value)

    @tf.function
    def train_step(targets, masks, sigma):
        """Make update the latent code `z` to make `G(z)` closer to `target`"""
        with tf.GradientTape() as tape:
            x = generate_image(z, c)
            errors = reconstruction_errors.blurred_mse(x, targets, masks, sigma)
            error = tf.reduce_mean(errors)
        gradients = tape.gradient(error, z)
        optimizer.apply_gradients(zip([gradients], [z]))
        return x, errors
    
# ============================= GENERATE TFRECORD ==============================
    
    # create directory for the tfrecord files
    os.makedirs(
        os.path.join(experiment_dir, 'trajectories', 'images', 'generated'), 
        exist_ok=True)
    os.makedirs(
        os.path.join(experiment_dir, 'trajectories', 'images', 'target'), 
        exist_ok=True)
    
    # create tfrecord writer
    writer = tf.io.TFRecordWriter(
        os.path.join(experiment_dir, 'trajectories', 'tfrecord.tfrecord')) 

    for batch in range(FLAGS.num_trajectories // FLAGS.batch_size):
        
        # Initialization
        z_0 = tf.random.normal([FLAGS.batch_size, z_dim])
        if FLAGS.renorm: 
            z_0 = tf.clip_by_norm(z_0, tf.math.sqrt(float(z_dim)), axes=[-1])
        z.assign(z_0)
        x_0 = generate_image(z, c)

        for t_sign in [-1, 1]:
            z.assign(z_0)
            for t_mod in np.linspace(0.1, 0.5, 5):
                t = t_sign * t_mod
                T.assign(t) 
                target, mask = transform(x_0, T)

                # Find z for the intermediate transformation
                initialize_optimizer()
                for step in range(FLAGS.n_steps):
                    x, errors = train_step(target, mask, FLAGS.sigma)
                    if FLAGS.renorm: renormalize(z)
                
                # Save images
                for i in range(FLAGS.batch_size):
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'z_0':   float_list_feature(z_0[i]),
                            'z_t':   float_list_feature(z[i]),
                            'y':     int64_feature(FLAGS.y),
                            'error':  float_feature(errors[i]),
                            'delta_t': float_feature(t),
                        }
                    ))
                    writer.write(example.SerializeToString())

                    save_images(x.numpy()[i:i+1,:,:,:], os.path.join(
                        experiment_dir, 'trajectories', 'images', 'generated',
                        '{}_t={:2.1f}.png'.format(batch*FLAGS.batch_size+i, t+0.5)))

                    save_images(target.numpy()[i:i+1,:,:,:], os.path.join(
                        experiment_dir, 'trajectories', 'images', 'target',
                        '{}_t={:2.1f}.png'.format(batch*FLAGS.batch_size+i, t+0.5))) 
    
    writer.close()

if __name__ == '__main__':
    # output directory
    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs', 'test'), 
        "output directory")
    
    # save images ?
    flags.DEFINE_bool('save_images', False, 
        "save images and tfrecord or just tfrecords")

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
    
    app.run(generate_trajectory)