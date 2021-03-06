
import numpy as np
import tensorflow as tf

from src.layer import *
import tensorflow.contrib.layers as layers

class LapSRN_v1_bak(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_srn'):
    self.scope = scope
    self.inputs = inputs
    self.gt_imgs = [gt_img_x2, gt_img_x4, gt_img_x8]
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])

    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.filter_num = filter_num
    self.reg = reg
    self.is_training = is_training
    self.residual_depth = 10
    self.kernel_size = 3
    self.image_squeeze_channle = 256
    self.batch_size, _, _, self.channel = tf.Tensor.get_shape(inputs).as_list()

    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def init_gt_imgs(self):
    pass

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for l in range(self.level):
        # current width and height for current stage.
        width = self.width*np.exp2(l).astype(int)
        height = self.height*np.exp2(l).astype(int)

        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}'.format(str(l), str(d)))

        # current upscaled width and height for current stage.
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        x = tf.image.resize_bilinear(x, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_transpose_upscale'.format(str(l)))
        # x = layers.conv2d_transpose(x, self.filter_num, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}__transpose_upscale'.format(str(l)))


        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(l)))

        self.extracted_features.append(net)

      base_images = self.inputs
      for l in range(self.level):
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        base_images = tf.image.resize_bilinear(base_images, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_biliear'.format(str(l)))
        # base_images = layers.conv2d_transpose(base_images, 1, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}_img_upscale_transpose'.format(str(l)))

        self.reconstructed_imgs.append(base_images)

  def extract_drrn_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for l in range(self.level):
        # current width and height for current stage.
        width = self.width*np.exp2(l).astype(int)
        height = self.height*np.exp2(l).astype(int)

        init = x
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_a'.format(str(l), str(d))):
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(l), str(d)))

          with tf.variable_scope('level_{}_residual_{}_b'.format(str(l), str(d))):
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(l), str(d)))

          x = init + x

        # current upscaled width and height for current stage.
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)

        with tf.variable_scope('level_{}_norm'.format(str(l))):
          x = batch_normalize(x, self.is_training)
          x = lrelu(x)

        x = tf.image.resize_bilinear(x, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_transpose_upscale'.format(str(l)))
        # x = layers.conv2d_transpose(x, self.filter_num, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}__transpose_upscale'.format(str(l)))


        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(l)))

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
    loss = []
    for l in range(self.level):
      loss.append(self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l]))

    loss = tf.reduce_sum(loss)

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
    loss = []
    for l in range(self.level):
      diff = tf.square(tf.subtract(self.sr_imgs[l], self.gt_imgs[l]))
      loss.append(0.5 * tf.reduce_mean(diff))

    return tf.reduce_sum(loss)

  def upscaled_img(self, index):
    return self.sr_imgs[np.log2(index).astype(int)-1]


class LapSRN_v1(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_srn'):
    self.scope = scope
    self.inputs = inputs
    self.gt_imgs = [gt_img_x2, gt_img_x4, gt_img_x8]
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])

    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.filter_num = filter_num
    self.reg = reg
    self.is_training = is_training
    self.residual_depth = 10
    self.kernel_size = 3
    self.image_squeeze_channle = 256
    self.batch_size, _, _, self.channel = tf.Tensor.get_shape(inputs).as_list()

    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def init_gt_imgs(self):
    pass

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      bilinear_weights_init = tf.constant_initializer(value=bilinear_upsample_weights(2, self.filter_num), dtype=tf.float32)

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for l in range(self.level):
        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}'.format(str(l), str(d)))

        # current upscaled width and height for current stage.
        x = layers.conv2d_transpose(x, self.filter_num, 4, stride=2, padding='SAME', activation_fn=lrelu, weights_initializer=bilinear_weights_init, biases_initializer=None, scope='level_{}__transpose_upscale'.format(str(l)))


        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(l)))

        self.extracted_features.append(net)

      base_images = self.inputs
      for l in range(self.level):
        upscaled_width = self.width*np.exp2(l+1).astype(int)
        upscaled_height = self.height*np.exp2(l+1).astype(int)
        # base_images = tf.image.resize_bilinear(base_images, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_biliear'.format(str(l)))

        bilinear_weights_init1 = tf.constant_initializer(value=bilinear_upsample_weights(2, 1), dtype=tf.float32)
        base_images = layers.conv2d_transpose(base_images, 1, 4, stride=2, padding='SAME', activation_fn=lrelu, weights_initializer=bilinear_weights_init1, biases_initializer=None, scope='level_{}_img_upscale_transpose'.format(str(l)))

        self.reconstructed_imgs.append(base_images)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.level):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def l1_loss(self):
    loss = []
    for l in range(self.level):
      loss.append(self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l]))

    loss = tf.reduce_sum(loss)

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
    loss = []
    for l in range(self.level):
      diff = tf.square(tf.subtract(self.sr_imgs[l], self.gt_imgs[l]))
      loss.append(0.5 * tf.reduce_mean(diff))

    return tf.reduce_sum(loss)

  def upscaled_img(self, index):
    return self.sr_imgs[-1]

class LapSRN_v2(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):
    self.scope = scope
    self.inputs = inputs
    self.labels = vars()["gt_img_x{}".format(upscale_factor)]
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])

    self.upscale_factor = upscale_factor
    self.level = np.log2(upscale_factor).astype(int)
    self.filter_num = filter_num
    self.reg = reg
    self.is_training = is_training
    self.residual_depth = 10
    self.kernel_size = 3
    self.batch_size, _, _, self.channel = tf.Tensor.get_shape(inputs).as_list()

    self.gt_imgs = []
    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def init_gt_imgs(self):
    for scale in range(1, self.upscale_factor):
      height = self.height*scale
      width = self.width*scale
      img = tf.image.resize_bicubic(self.labels, size=[height, width], align_corners=False, name='level_{}_gt_img'.format(str(scale)))
      self.gt_imgs.append(img)
    self.gt_imgs.append(self.labels)

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        # x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for scale in range(1, self.upscale_factor+1):

        with tf.variable_scope('level_{}_img'.format(str(scale))):
          # current width and height for current stage.
          height = self.height*scale
          width = self.width*scale
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(scale)))
          # x = layers.conv2d_transpose(x, self.filter_num, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, , weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}__transpose_upscale'.format(str(l)))
          # x = deconv_layer(x, [3, 3, self.filter_num, self.filter_num], [self.batch_size, height, width, self.filter_num], stride=1)

        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}'.format(str(scale), str(d)))

        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(scale)))

        self.extracted_features.append(net)

      base_images = self.inputs
      for scale in range(1, self.upscale_factor+1):
        upscaled_width = self.width*scale
        upscaled_height = self.height*scale
        base_images = tf.image.resize_bilinear(base_images, size=[upscaled_height, upscaled_width], align_corners=False, name='level_{}_biliear'.format(str(scale)))

        # base_images = layers.conv2d_transpose(base_images, 1, 4, stride=2, padding='SAME', activation_fn=lrelu, biases_initializer=None, scope='level_{}_img_upscale_transpose'.format(str(l)))
        # net = deconv_layer(x, [self.kernel_size, self.kernel_size, self.channel, self.filter_num], [self.batch_size, upscaled_height, upscaled_width, self.channel], stride=1)

        self.reconstructed_imgs.append(base_images)

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.upscale_factor):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def l1_loss(self):
    if len(self.gt_imgs) == 0:
      self.init_gt_imgs()

    loss = 0.0
    for l in range(self.upscale_factor):
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

  def upscaled_img(self, index):
    return self.sr_imgs[index-1]

class BaselapV1(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='lap_ml_srn'):
    self.scope = scope
    self.inputs = inputs
    self.labels = vars()["gt_img_x{}".format(upscale_factor)]
    self.batch_size, _, _, self.channel = tf.Tensor.get_shape(inputs).as_list()
    self.height = np.uint32(image_size[0])
    self.width = np.uint32(image_size[1])
    self.upscale_factor = upscale_factor
    self.filter_num = filter_num
    self.reg = reg
    self.is_training = is_training

    # hyper parameters
    self.step_depth = 4
    self.residual_depth = 10
    self.kernel_size = 3

    self.gt_imgs = []
    self.sr_imgs = []
    self.reconstructed_imgs = []
    self.extracted_features = []

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def init_gt_imgs(self):
    for step in range(1, self.step_depth):
      height = self.height*self.upscale_factor*step//self.step_depth
      width = self.width*self.upscale_factor*step//self.step_depth
      img = tf.image.resize_bicubic(self.labels, size=[height, width], align_corners=False, name='level_{}_gt_img'.format(str(step)))
      self.gt_imgs.append(img)

    self.labels.set_shape([self.batch_size, self.height*self.upscale_factor, self.width*self.upscale_factor, self.channel])
    self.gt_imgs.append(self.labels)

  def current_step_img_size(self, step):
    current_tensor = self.gt_imgs[step]
    _, current_height, current_width, _ = tf.Tensor.get_shape(current_tensor).as_list()

    return current_height, current_width

  def extract_features(self, reuse=False):
    pass

  def reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.step_depth):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def l1_normal_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

  def l1_weighted_loss(self, base=1):
    loss = 0.0
    for l in range(self.step_depth):
      loss = loss + ((l+base)/self.step_depth) * self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

  def l1_loss(self):
    return self.l1_normal_loss()

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

  def upscaled_img(self, index):
    # r = int(index * self.step_depth / self.upscale_factor - 1)
    return self.sr_imgs[-1]

  def get_image(self, index):
    # r = int(index * self.step_depth / self.upscale_factor - 1)
    return self.sr_imgs[-1]

class LapSRN_v3(BaselapV1):

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)

        net = layers.conv2d(x, 1, kernel_size=5, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def l1_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      # loss = loss + (1.0/self.step_depth*l) * self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])
      loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

class LapSRN_v4(BaselapV1):

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, 1024, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          # net = batch_normalize(net, self.is_training)
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def l1_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])
      # loss = loss + (1.0/self.step_depth*l) * self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

class LapSRN_v5(BaselapV1):

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        # x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
            # x = batch_normalize(x, self.is_training)

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, 512, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          # net = batch_normalize(net, self.is_training)
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def l1_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      # loss = loss + self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])
      loss = loss + (1.0*l/self.step_depth) * self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])

    return loss

class LapSRN_v6(BaselapV1):
  '''
    drrn image_tune: 512x1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
        # x = batch_normalize(x, self.is_training)
        x = lrelu(x)

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        init = x
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_a'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))

          with tf.variable_scope('level_{}_residual_{}_b'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))

          x = init + x

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, self.image_squeeze_channle, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          # net = batch_normalize(net, self.is_training)
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)


class LapSRN_v7(LapSRN_v6):
  '''
    drrn image_tune: 1024x1 residual_depth = 12
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 12
    self.image_squeeze_channle = 1024

class LapSRN_v8(LapSRN_v6):
  '''
    drrn image_squeeze_channle: 256 residual_depth = 10
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 256

class LapSRN_v9(LapSRN_v6):
  '''
    drrn image_squeeze_channle: 256 residual_depth = 10 l1_weighted_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 256

  def l1_loss(self):
    return self.l1_weighted_loss(3)

class LapSRN_v10(LapSRN_v6):
  '''
    drrn image_tune: 512x1 step_depth:4 residual_depth:10 weighted_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

  def l1_loss(self):
    return self.l1_weighted_loss(3)

class LapSRN_v11(LapSRN_v6):
  '''
    drrn image_tune: 1024 step_depth:4 residual_depth:14
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 14
    self.image_squeeze_channle = 1024

  def l1_loss(self):
    return self.l1_weighted_loss(3)


class LapSRN_v12(LapSRN_v6):
  '''
    drrn image_tune: 1024 step_depth:4 residual_depth:10 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 1024

class LapSRN_v13(LapSRN_v6):
  '''
    drrn image_tune: 1024x1 step_depth:4 residual_depth:16 weighted_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 64

class LapSRN_v14(LapSRN_v6):
  '''
    drrn image_tune: 1024x1 step_depth:4 residual_depth:12 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 128

class LapSRN_v15(LapSRN_v6):
  '''
    drrn image_tune: 1024x1 step_depth:2 residual_depth:12 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512

class LapSRN_v16(LapSRN_v6):
  '''
    drrn image_tune: 1024x1 step_depth:2 residual_depth:12 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 5
    self.kernel_size = 3
    self.residual_depth = 4
    self.image_squeeze_channle = 512

class LapSRN_v17(LapSRN_v6):
  '''
    drrn image_tune: 512 step_depth:2 residual_depth:8 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 10
    self.kernel_size = 3
    self.residual_depth = 2
    self.image_squeeze_channle = 512

class LapSRN_v18(LapSRN_v6):
  '''
    drrn image_tune: 512 step_depth:2 residual_depth:8 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512



class LapSRN_v19(LapSRN_v6):
  '''
    drrn image_tune: 512 step_depth:1 residual_depth:20 normal_loss
  '''

  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 20
    self.image_squeeze_channle = 512


class LapSRN_v2_v1(BaselapV1):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:10 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 1024

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('step_{}_init'.format(step)):
          x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

        init = x
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_a'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))
          with tf.variable_scope('level_{}_residual_{}_b'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))
          x = init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, self.image_squeeze_channle, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          # net = batch_normalize(net, self.is_training)
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)

class LapSRN_v2_v2(BaselapV1):
  '''
    drrn image_tune: 512x1 step_depth:4 residual_depth:10 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 1024

  def extract_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for step in range(1, self.step_depth+1):

        with tf.variable_scope('step_{}_init'.format(step)):
          x = layers.conv2d(self.inputs, self.filter_num*step, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

        init = x
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_a'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num*step, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))
          with tf.variable_scope('level_{}_residual_{}_b'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num*step, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))
          x = init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, self.image_squeeze_channle, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          # net = batch_normalize(net, self.is_training)
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)

# upscale 2
class LapSRN_v30(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:2 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

class LapSRN_v31(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512


class LapSRN_v32(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:6 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

class LapSRN_v33(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:8 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 5
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

class LapSRN_v34(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:10 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512

# upscale 8
class LapSRN_v40(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:4 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 6
    self.image_squeeze_channle = 1024

class LapSRN_v41(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:6 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 30
    self.image_squeeze_channle = 1024

class LapSRN_v42(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:8 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 15
    self.image_squeeze_channle = 1024

class LapSRN_v43(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:10 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 1024

class LapSRN_v44(LapSRN_v6):
  '''
  drrn image_tune: 512x1 step_depth:4 residual_depth:12 normal_loss
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=1e-4, scope='lap_ml_srn'):

    BaselapV1.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    # hyper parameters
    self.step_depth = 10
    self.kernel_size = 3
    self.residual_depth = 6
    self.image_squeeze_channle = 1024

# update new visions for SR models

