# Train a model to predict the value of the parameter `t` of factor of variation
# from the latent code of the image.

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K 

from absl import app, flags, logging

from utils.lattice_layer import Lattice

z_dim = 120 # latent space dimension
n_cat = 1000 # number of categories

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, "")
flags.DEFINE_integer('batch_size_II', 32, "")

def load(tfrecords, batch_size=1, shuffle=False):
    def _decode(example):
        features = tf.io.parse_single_example(example, features={
            'z_0':   tf.io.FixedLenFeature(shape=[z_dim], dtype=tf.float32),
            'z_t':   tf.io.FixedLenFeature(shape=[z_dim], dtype=tf.float32),
            'y':     tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            'error':  tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
            'delta_t': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        })
        return features
    
    def convert(features):
        label = features['delta_t']
        features = [features['z_0'], features['z_t']]
        return features, label

    dataset = tf.data.TFRecordDataset(tfrecords) 
    dataset = dataset.shuffle(1100) if shuffle else dataset
    dataset = dataset.map(_decode, num_parallel_calls=8)

    # ============ Remove samples with error within the last decile ============
    errors = []
    for features in dataset:
        errors.append(features['error'].numpy())
    errors = np.sort(errors)
    error_threshold = errors[9 * errors.shape[0] // 10] 
    dataset = dataset.filter(
        lambda x: tf.math.greater(error_threshold, x['error']))
    # ==========================================================================
    
    dataset = dataset.map(convert, num_parallel_calls=8)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)
    
    return dataset

def learn_direction():    
    
    experiment_dir = os.path.join(
        FLAGS.output_dir, 
        '{}'.format(FLAGS.transform), 
        '{}'.format(FLAGS.y))

# ============================== LOAD DATASET ==================================

    tfrecord_path = [os.path.join(
        experiment_dir, 'trajectories', 'tfrecord.tfrecord')]
    
    dataset = load(tfrecord_path, batch_size=FLAGS.batch_size_II, shuffle=True)

# ====================== BUILD AND TRAIN LATENT MODEL ==========================

    # model of the form t = g(<u, z>)
    position_model = K.Sequential([
        # scalar product
        K.layers.Dense(1, 
            input_shape=[z_dim], use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
            name='scalar_product'),
        # g increasing piecewise-linear function 
        # (see module for implementation details)
        Lattice(regularizer_amount=1.0)
    ])

    # model of the form dt = g(<u, z_1>) - g(<u, z_0>)
    def delta_model():
        z0_z1 = K.Input([2,z_dim])
        z0 = K.layers.Lambda(lambda x: x[:,0,:], output_shape=[z_dim])(z0_z1)
        z1 = K.layers.Lambda(lambda x: x[:,1,:], output_shape=[z_dim])(z0_z1)
        t0 = position_model(z0)
        t1 = position_model(z1)
        dt = K.layers.Subtract()([t1, t0])
        return K.Model(inputs=z0_z1, outputs=dt)

    model = delta_model()
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae', 'mse'])
    model.fit(dataset, epochs=FLAGS.epochs, verbose=0)

# ======================= SAVE DIRECTION IN A CSV FILE =========================

    # get the main direction   
    v = model.trainable_variables[0]
    w = v / tf.linalg.norm(v)
    w = w.numpy()
    u = v / tf.square(tf.linalg.norm(v))
    u = u.numpy()

    os.makedirs(os.path.join(experiment_dir, 'directions'), exist_ok=True)
    filename = os.path.join(experiment_dir, 'directions', 'direction.csv')
    
    with open(filename, 'w') as f:
        f.write(', '.join(["{0:7.6f}".format(e[0]) for e in w]))

    # # visualization
    # import matplotlib.pyplot as plt
    # t = [np.linspace(-5, 5, 100)]

    # X = tf.matmul(tf.transpose(tf.cast(t, tf.float32)), tf.transpose(w))
    # Y = position_model(X).numpy().reshape(100)

    # plt.plot(t[0], Y-Y[50], c='r') # center the plot

    # X = []
    # Y = []
    # for features, labels in dataset:
    #     for x in tf.matmul(features[:,1,:] - features[:,0,:], w):
    #         X.append(x.numpy())
    #     for y in labels.numpy():#position_model(features[:,1,:]):
    #         Y.append(y)
    # Y = np.array(Y)
    # print(Y)
    # print(Y.shape)
    # X = np.concatenate(X)
    # # Y = np.concatenate(Y)

    # plt.scatter(X, Y)
    # plt.savefig(os.path.join(experiment_dir, 'fig_test.png'))

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
    
    app.run(learn_direction)