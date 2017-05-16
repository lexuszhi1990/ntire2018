import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from src.ops import *

class LapSRN(object):
  def __init__(self, inputs, gt_imgs, upscale_factor=4, filter_num=64, image_size = [64, 64]):
    self.scope = 'lap_srn'
    self.filter_num = filter_num
    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.residual_depth = 10
    self.kernel_size = 3
    self.height = image_size[0]
    self.width = image_size[1]

    self.inputs = inputs
    self.gt_imgs = gt_imgs

    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      net = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, padding='SAME', activation_fn=lrelu, scope='init')

      for l in xrange(self.level):
        for d in range(self.residual_depth):
          net = layers.conv2d(net, self.filter_num, kernel_size=self.kernel_size, scope='level_{}_residual_{}'.format(str(l), str(d)))
        net = layers.conv2d_transpose(net, self.filter_num, kernel_size=self.kernel_size, stride=2, padding='VALID', activation_fn=lrelu, scope='level_{}_transpose'.format(str(l)))
        net = net[:,0:-1, 0:-1, :]

        net = layers.conv2d(net, 3, kernel_size=self.kernel_size, padding='SAME', activation_fn=None, scope='level_{}_features'.format(str(l)))
        self.extracted_features.append(net)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      base_images = self.inputs
      for l in xrange(self.level):
        base_images = tf.image.resize_bilinear(base_images, size=[self.height*np.exp2(l+1).astype(int), self.width*np.exp2(l+1).astype(int)], align_corners=True, name='level_{}_biliear'.format(str(l)))
        self.reconstructed_imgs.append(base_images)

      for l in xrange(self.level):
        self.sr_imgs.append(tf.nn.tanh(self.reconstructed_imgs[l] + self.extracted_features[l]))

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def l1_charbonnier_loss(self):
    eps = 1e-6
    diff = tf.subtract(self.gt_imgs, self.sr_imgs[-1])
    error = tf.sqrt( diff * diff + eps)
    loss  = tf.reduce_mean(error)

    return loss

  def l2_loss(self):
    # diff = (X - Z) .^ 2;
    # Y = 0.5 * sum(diff(:));
    diff = tf.square(tf.subtract(self.gt_imgs, self.sr_imgs[-1]))

    return 0.5 * tf.reduce_mean(diff)
