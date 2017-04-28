# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib

import scipy.misc

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    return tf.maximum(x, leak*x)

def leaky_relu_batch_norm(x, alpha=0.2, name="lrelu_bn"):
  with tf.variable_scope(name):
    return lrelu(layers_lib.batch_norm(x), alpha)

def relu_batch_norm(x, name="relu_bn"):
  with tf.variable_scope(name):
    return tf.nn.relu(layers_lib.batch_norm(x))

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss

def content_loss(endpoints_dict, content_layers):
    content_loss = 0.0
    for layer in content_layers:
        generated_images, content_images = tf.split(axis=0, num_or_size_splits=2, value=endpoints_dict[layer])
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss / len(content_layers)

def im_resize(x, fraction, interp='bilinear'):
  '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html
    interp : str, optional
    Interpolation to use for re-sizing (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
  '''
  pass
