import os
import tensorflow as tf

from absl import app, flags, logging
from numba import cuda

from I_generate_trajectory import generate_trajectory
from II_learn_direction import learn_direction
from III_latent_traversal import latent_traversal
from IV_get_barycenter import get_barycenter
from V_measure_perf import measure_perf



gpus = tf.config.experimental.list_physical_devices('ǴPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    
def main(argv):
    generate_trajectory()
    learn_direction()

    if FLAGS.evaluation in ['qualitative', 'quantitative']:
        latent_traversal()
    if FLAGS.evaluation == 'quantitative':
        cuda.select_device(0)
        cuda.close()
        get_barycenter()
        measure_perf()

if __name__ == '__main__':
    FLAGS = flags.FLAGS

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
    
    # evaluation
    flags.DEFINE_enum('evaluation', 'none', 
        ['none', 'qualitative', 'quantitative'], 
        "type of evaluation")

    app.run(main)
