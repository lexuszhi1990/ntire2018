import numpy as np
import tensorflow as tf

from src.layer import *
import tensorflow.contrib.layers as layers

class LapSRN(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, image_size, is_training, upscale_factor=4, filter_num=64, scope='lap_srn'):
    self.scope = scope
    self.inputs = inputs
    self.gt_imgs = [gt_img_x2, gt_img_x4]
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])

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

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        # x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for l in range(self.level):
        # current width and height for current stage.
        width = self.width*np.exp2(l).astype(int)
        height = self.height*np.exp2(l).astype(int)

        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}'.format(str(l), str(d))):
            x = deconv_layer(x, [self.kernel_size, self.kernel_size, self.filter_num, self.filter_num], [self.batch_size, height, width, self.filter_num], stride=1)
            # x = batch_normalize(x, self.is_training)
            x = lrelu(x)

        # current upscaled width and height for current stage.
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        with tf.variable_scope('level_{}_pixel_shift_upscale'.format(str(l))):
          x = deconv_layer(x, [self.kernel_size, self.kernel_size, self.filter_num*4, self.filter_num], [self.batch_size, height, width, self.filter_num*4], stride=1)
          x = pixel_shuffle_layer(x, 2, 64)
          x = lrelu(x)
        # with tf.variable_scope('level_{}_transpose_upscale'.format(str(l))):
          # x = deconv_layer(x, [4, 4, self.filter_num, self.filter_num], [self.batch_size, upscaled_height, upscaled_width, self.filter_num], stride=2)
          # x = batch_normalize(x, self.is_training)
          # x = lrelu(x)

        with tf.variable_scope('level_{}_img'.format(str(l))):
          net = deconv_layer(x, [self.kernel_size, self.kernel_size, self.channel, self.filter_num], [self.batch_size, upscaled_height, upscaled_width, self.channel], stride=1)

          self.extracted_features.append(net)

      base_images = self.inputs
      for l in range(self.level):
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)
        with tf.variable_scope('level_{}_img_upscale_transpose'.format(str(l))):
          base_images = deconv_layer(base_images, [4, 4, self.channel, self.channel], [self.batch_size, upscaled_height, upscaled_width, self.channel], stride=2)
          # base_images = tf.image.resize_bilinear(base_images, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_biliear'.format(str(l)))

        self.reconstructed_imgs.append(base_images)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.level):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def l1_loss(self):
    loss = 0.0
    for l in range(self.level):
      loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

  def l1_charbonnier_loss(self, X, Y):
    eps = 1e-6
    diff = tf.subtract(X, Y)
    error = tf.sqrt( diff * diff + eps)
    loss  = tf.reduce_mean(error)

    return loss

  def l2_loss(self):
    # diff = (X - Z) .^ 2;
    # Y = 0.5 * sum(diff(:));
    diff = tf.square(tf.subtract(self.gt_imgs[-1], self.sr_imgs[-1]))

    return 0.5 * tf.reduce_mean(diff)

class LapSRN_v1(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, image_size, is_training, upscale_factor=4, filter_num=64, scope='lap_srn'):
    self.scope = scope
    self.inputs = inputs
    self.gt_imgs = [gt_img_x2, gt_img_x4]
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])

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

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='init')

      for l in range(self.level):
        # current width and height for current stage.
        width = self.width*np.exp2(l).astype(int)
        height = self.height*np.exp2(l).astype(int)

        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}_residual_{}'.format(str(l), str(d)))

        # current upscaled width and height for current stage.
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        x = tf.image.resize_bilinear(x, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_transpose_upscale'.format(str(l)))
        # x = layers.conv2d_transpose(x, self.filter_num, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}__transpose_upscale'.format(str(l)))


        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}_img'.format(str(l)))

        self.extracted_features.append(net)

      base_images = self.inputs
      for l in range(self.level):
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        base_images = tf.image.resize_bilinear(base_images, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_biliear'.format(str(l)))
        # base_images = layers.conv2d_transpose(base_images, 1, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}_img_upscale_transpose'.format(str(l)))

        self.reconstructed_imgs.append(base_images)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.level):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def l1_loss(self):
    loss = 0.0
    for l in range(self.level):
      loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

  def l1_charbonnier_loss(self, X, Y):
    eps = 1e-6
    diff = tf.subtract(X, Y)
    error = tf.sqrt( diff * diff + eps)
    loss  = tf.reduce_mean(error)

    return loss

  def l2_loss(self):
    # diff = (X - Z) .^ 2;
    # Y = 0.5 * sum(diff(:));
    loss = 0.0
    for l in range(self.level):
      diff = tf.square(tf.subtract(self.sr_imgs[l], self.gt_imgs[l]))
      loss = loss + 0.5 * tf.reduce_mean(diff)

    return loss
