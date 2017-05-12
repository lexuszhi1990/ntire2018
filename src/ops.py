# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    return tf.maximum(x, leak*x)

def leaky_relu_batch_norm(x, alpha=0.2, name="lrelu_bn"):
  with tf.variable_scope(name):
    return lrelu(layers_lib.batch_norm(x), alpha)

def relu_batch_norm(x, name="relu_bn"):
  with tf.variable_scope(name):
    return tf.nn.relu(layers_lib.batch_norm(x))
