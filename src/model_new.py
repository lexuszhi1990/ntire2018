import numpy as np
import tensorflow as tf

from src.layer import *
import tensorflow.contrib.layers as layers

# abstract model for sr
class BaseModel(object):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='edsr'):
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
    self.residual_depth = 4
    self.kernel_size = 3
    self.image_squeeze_channle = 256

    # results container
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
    pass

  def residual_reconstruct(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      for l in range(self.step_depth):
        sr_img = tf.add(self.reconstructed_imgs[l], self.extracted_features[l])
        self.sr_imgs.append(sr_img)

  def normal_reconstruct(self, reuse=False):
    self.sr_imgs = self.extracted_features

  def extract_baseline_features_without_BN(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for step in range(1, self.step_depth+1):
        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_bilinear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def extract_baseline_features_with_BN(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for step in range(1, self.step_depth+1):
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_convA'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)

          with tf.variable_scope('level_{}_residual_{}_convB'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_bilinear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def extract_recurrence_features_without_BN(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for step in range(1, self.step_depth+1):
        init = x
        for d in range(self.residual_depth):
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))
          x = init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_bilinear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def extract_ed_block_features(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      with tf.variable_scope('init'):
        x = deconv_layer(self.inputs, [self.kernel_size, self.kernel_size, self.filter_num, self.channel], [self.batch_size, self.height, self.width, self.filter_num], stride=1)
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
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)

  def l1_loss(self):
    loss = []
    for l in range(self.step_depth):
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
    for l in range(self.step_depth):
      diff = tf.square(tf.subtract(self.sr_imgs[l], self.gt_imgs[l]))
      loss.append(0.5 * tf.reduce_mean(diff))

    return tf.reduce_sum(loss)

  def upscaled_img(self, index):
    return self.sr_imgs[-1]

# baseline learning for without BN sr
class EDSR_v100(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_baseline_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.normal_reconstruct(reuse)

# baseline learning for with BN
class EDSR_v101(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_baseline_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.normal_reconstruct(reuse)

# residual learning without BN
class EDSR_v102(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_baseline_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

# residual learning with BN for sr
class EDSR_v103(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_baseline_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

# recurrence learning without BN for sr
class EDSR_v104(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.normal_reconstruct(reuse)

# recurrence residual learning without BN
class EDSR_v105(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

class EDSR_v301(BaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5

  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)
