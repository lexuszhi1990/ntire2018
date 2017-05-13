import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from src.ops import *

class LapSRN(object):
  def __init__(self, upscale_factor=4, filter_num=64, is_training=True):
    self.scope = 'lap_srn'
    self.filter_num = filter_num
    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.residual_depth = 10
    self.kernel_size = 3
    self.width = 64
    self.height = 64

    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  def forward(self, inputs, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      net = layers.conv2d(inputs, self.filter_num, kernel_size=self.kernel_size, padding='SAME', activation_fn=lrelu, scope='init')

      for l in xrange(self.level):
        for d in range(self.residual_depth):
          net = layers.conv2d(net, self.filter_num, kernel_size=self.kernel_size, scope='level_{}_residual_{}'.format(str(l), str(d)))
        net = layers.conv2d_transpose(net, filter_num, kernel_size=kernel_size, stride=2, padding='VALID', activation_fn=lrelu, scope='level_{}_transpose'.format(str(level)))
        net = net[:,0:-1, 0:-1, :]

        net = layers.conv2d(net, 3, kernel_size=self.kernel_size, padding='SAME', activation_fn=None, scope='level_{}_features'.format(str(level)))
        self.extracted_features.append(net)

      base_images = inputs
      for l in xrange(self.level):
        base_images = tf.image.resize_bilinear(base_images, size=[self.height*np.exp2(l+1).astype(int), self.width*np.exp2(l+1).astype(int)], align_corners=True, name='level_{}_biliear'.format(str(l)))
        self.reconstructed_imgs.append(base_images)

      for l in xrange(self.level):
        self.sr_imgs.append(tf.nn.tanh(self.reconstructed_imgs[l] + self.extracted_features[l]))

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def l1_charbonnier_loss(self, gt_imgs):
    eps = 1e-6
    error = tf.sqrt(tf.add(tf.square(tf.subtract(gt_imgs, self.sr_imgs[-1])), eps))

    return tf.reduce_sum(error)

  def l2_loss(self, gt_imgs):
    # diff = (X - Z) .^ 2;
    # Y = 0.5 * sum(diff(:));
    diff = tf.square(tf.subtract(gt_imgs, self.sr_imgs[-1]))
    error = 0.5 * tf.reduce_sum(diff)

    return error
