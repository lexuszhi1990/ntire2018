import numpy as np
import tensorflow as tf

from src.layer import *

class LapSRN(object):
  def __init__(self, inputs, gt_imgs, image_size, is_training, upscale_factor=4, filter_num=64, scope='lap_srn'):
    self.scope = scope
    self.inputs = inputs
    self.gt_imgs = gt_imgs
    self.height = image_size[0]
    self.width = image_size[1]

    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.filter_num = filter_num
    self.is_training = is_training
    self.residual_depth = 10
    self.kernel_size = 3
    self.batch_size, _, _, self.channel = tf.Tensor.get_shape(inputs).as_list()

    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        x = prelu(x)

      for l in xrange(self.level):
        # current width and height for current stage.
        width = self.width*np.exp2(l).astype(int)
        height = self.height*np.exp2(l).astype(int)

        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}'.format(str(l), str(d))):
            x = deconv_layer(x, [self.kernel_size, self.kernel_size, self.filter_num, self.filter_num], [self.batch_size, height, width, self.filter_num], stride=1)
            x = batch_normalize(x, self.is_training)
            x = tf.nn.relu(x)

        # current upscaled width and height for current stage.
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        with tf.variable_scope('level_{}_pixel_shift_upscale'.format(str(l))):
          x = deconv_layer(x, [self.kernel_size, self.kernel_size, self.filter_num*4, self.filter_num], [self.batch_size, height, width, self.filter_num*4], stride=1)
          x = pixel_shuffle_layer(x, 2, 64)
          x = tf.nn.relu(x)

        with tf.variable_scope('level_{}_img'.format(str(l))):
          net = deconv_layer(x, [self.kernel_size, self.kernel_size, self.channel, self.filter_num], [self.batch_size, upscaled_height, upscaled_width, self.channel], stride=1)
        self.extracted_features.append(net)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      base_images = self.inputs
      for l in xrange(self.level):
        base_images = tf.image.resize_bicubic(base_images, size=[self.height*np.exp2(l+1).astype(int), self.width*np.exp2(l+1).astype(int)], align_corners=False, name='level_{}_biliear'.format(str(l)))
        self.reconstructed_imgs.append(base_images)

      for l in xrange(self.level):
        # self.sr_imgs.append(tf.nn.tanh(self.reconstructed_imgs[l] + self.extracted_features[l]))
        self.sr_imgs.append(self.reconstructed_imgs[l] + self.extracted_features[l])

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
