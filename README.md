# Controlling generative models with continuous factors of variations

Official repository of the paper Plumerault et. al. (2019)
(https://openreview.net/forum?id=H1laeJrKDB)

### abstract

Recent deep generative models are able to provide photo-realistic images as well as visual or textual content embeddings useful to address various tasks of computer vision and natural language processing. Their usefulness is nevertheless often limited by the lack of control over the generative process or the poor understanding of the learned representation. To overcome these major issues, very recent work has shown the interest of studying the semantics of the latent space of generative models. In this paper, we propose to advance on the interpretability of the latent space of generative models by introducing a new method to find meaningful directions in the latent space of any generative model along which we can move to control precisely specific properties of the generated image like the position or scale of the object in the image. Our method does not require human annotations and is particularly well suited for the search of directions encoding simple transformations of the generated image, such as translation, zoom or color variations. We demonstrate the effectiveness of our method qualitatively and quantitatively, both for GANs and variational auto-encoders.

## Dependencies

* tensorflow 2
* pytorch
* matplotlib
* opencv
* numba
* (tensorflow hub)

For easy setup the project includes a Dockerfile and a requirements.txt file.

## Models and weights

GAN model weights are taken from tensorflow-hub and have been converted when necessary to be compatible with tensorflow 2. Saliency model is taken from the official pytorch repository of the paper describing the model Hou et. al. (2017) (https://github.com/Andrew-Qibin/PoolNet). 

## Overview

This work aims at finding meaningfull directions in the latent space of generative models along which we can move to control precisely specific properties of the generated image like the position or scale of the object in the image. Our method does not require human annotations and is particularly well suited for the search of directions encoding simple transformations of the generated image, such as translation, zoom or color variations.

## Usage

the folder scripts contain scripts to run full experiment with sensible hyperparameters. To run the full procedure, simply use the file `src/main.py`. You can also run parts of the method like generate trajectory or learn the latent space model by using the appropriate file. Only vertical position, horizontal position, scale and brightness are implemented. Adding your own transformation requires writing code in `src/utils/transformation.py` and modifying slightly the other files.
