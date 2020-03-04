# Save figures to measure performance

import os
import numpy as np 
import matplotlib.pyplot as plt

from absl import app, flags, logging
from matplotlib import rc

from utils.ILSVRC_dict import number_to_name

FLAGS = flags.FLAGS

plt.style.use('seaborn')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

def measure_perf():

    experiment_dir = os.path.join(
        FLAGS.output_dir, 
        '{}'.format(FLAGS.transform), 
        '{}'.format(FLAGS.y))

    os.makedirs(os.path.join(experiment_dir, 'figures'), exist_ok=True)

# ============================== READ MEASURMENTS ==============================
    
    filename = os.path.join(experiment_dir, 'barycenters', 'barycenters.csv')
    with open(filename, 'r') as f:
        f.readline()
        T = [[] for _ in range(11)]
        for i, line in enumerate(f):
            line = line.split(',')
            if float(line[3]) > 0.02: # saliency detection fails
                if FLAGS.transform == 'horizontal_position':
                    T[i%11].append(float(line[1]))
                if FLAGS.transform == 'vertical_position':
                    T[i%11].append(float(line[2]))
                if FLAGS.transform == 'scale':
                    T[i%11].append(float(line[3]))
    
    # make it increase
    a = np.median(T[0])
    b = np.median(T[-1])
    if (a > b):
        T.reverse()

    # compute standard deviation
    std = [np.std(t) for t in T]
    
# ============================= SINGLE VIOLIN PLOT =============================
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(
        r'\textbf{{category:}} {} \textbf{{factor of variation:}} {}'.format(
            number_to_name[FLAGS.y].split(',')[0], 
            FLAGS.transform.replace('_', ' ')))
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'\textbf{{{}}}'.format(FLAGS.transform.replace('_', ' ')))
    ax.set_ylim([0.0, 1.0] if FLAGS.transform == 'scale' else [-0.4, 0.4])
    ax.set_xticklabels(['{:2.1f}'.format(e) for e in np.linspace(-5, 5, 11)])
    ax.violinplot(T, showextrema=False)
    ax.boxplot(T, showfliers=False)
    # ax.set_aspect(10 if FLAGS.transform == 'scale' else 10/0.8)
    ax2 = ax.twinx()
    ax2.set_ylim([0.0, 1.0] if FLAGS.transform == 'scale' else [0, 0.8])
    ax2.set_ylabel(r'\textbf{standard deviation}')
    ax2.plot(range(1,12), std, c='r')
    
    
    # ax2.set_aspect(10 if FLAGS.transform == 'scale' else 10/0.8)
    ax2.grid(False)
    fig.savefig(os.path.join(
        experiment_dir, 'figures', 'violinplot.png'), bbox_inches='tight') 


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
    
    app.run(get_barycenter)