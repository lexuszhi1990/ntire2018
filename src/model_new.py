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
    self.image_g_kernel_size = 3

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
          local_init = x
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convA'.format(str(step), str(d)))
          x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg), scope='level_{}_residual_{}_convB'.format(str(step), str(d)))
          x = local_init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
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
          local_init = x
          with tf.variable_scope('level_{}_residual_{}_convA'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)

          with tf.variable_scope('level_{}_residual_{}_convB'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
          x = local_init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
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

        net = layers.conv2d(x, 1, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_bilinear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def extract_recurrence_features_with_BN(self, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      x = layers.conv2d(self.inputs, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='init')

      for step in range(1, self.step_depth+1):
        init = x
        for d in range(self.residual_depth):
          with tf.variable_scope('level_{}_residual_{}_convA'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
          with tf.variable_scope('level_{}_residual_{}_convB'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
          x = init + x

        with tf.variable_scope('level_{}_img'.format(str(step))):
          height, width = self.current_step_img_size(step-1)
          x = tf.image.resize_bilinear(x, size=[height, width], align_corners=False, name='level_{}_transpose_upscale'.format(str(step)))

        net = layers.conv2d(x, 1, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg), scope='level_{}_img'.format(str(step)))
        self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_bilinear'.format(str(step)))

        self.reconstructed_imgs.append(base_images)

  def extract_ed_block_features_without_BN(self, reuse=False):
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
          net = layers.conv2d(x, self.image_squeeze_channle, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)

  def extract_ed_block_features_with_BN(self, reuse=False):
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
          with tf.variable_scope('level_{}_residual_{}_convA'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
          with tf.variable_scope('level_{}_residual_{}_convB'.format(str(step), str(d))):
            x = layers.conv2d(x, self.filter_num, kernel_size=self.kernel_size, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer = layers.l2_regularizer(scale=self.reg))
            x = batch_normalize(x, self.is_training)
            x = lrelu(x)
          local_init = x
          x = local_init + x
        x = init + x

        with tf.variable_scope('level_{}_upscaled_img'.format(str(step))):
          net = layers.conv2d(x, self.image_squeeze_channle, kernel_size=self.image_g_kernel_size, stride=1, padding='SAME', activation_fn=lrelu, biases_initializer=None, weights_regularizer=layers.l2_regularizer(scale=self.reg))
          net = layers.conv2d(net, 1, kernel_size=1, stride=1, padding='SAME', activation_fn=None, biases_initializer=None, weights_regularizer=None, scope='level_{}_img'.format(str(step)))
          self.extracted_features.append(net)

      base_images = self.inputs
      for step in range(1, self.step_depth+1):
        height, width = self.current_step_img_size(step-1)
        base_images = tf.image.resize_bilinear(base_images, size=[height, width], align_corners=False, name='level_{}_biliear'.format(str(step)))
        self.reconstructed_imgs.append(base_images)

  def total_variation_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      # loss = loss + total_variation_loss(self.sr_imgs[l])
      loss = loss + tf.reduce_sum(tf.image.total_variation(self.sr_imgs[l]))

    return loss

  def weighted_loss(self):
    loss = []
    base = (self.step_depth-1)/2
    for l in range(self.step_depth):
      step_loss = (l+base)/float(self.step_depth)*self.l1_charbonnier_loss(self.sr_imgs[l], self.gt_imgs[l])
      loss.append(step_loss)

    loss = tf.reduce_sum(loss)
    return loss

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

  def get_image(self, index):
    return self.sr_imgs[(index*self.step_depth)//self.upscale_factor - 1]

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

# recurrence learning with BN for sr
class EDSR_v107(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_recurrence_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.normal_reconstruct(reuse)

# recurrence residual learning without BN
class EDSR_v105(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

# recurrence residual learning with BN
class EDSR_v106(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_recurrence_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

# recurrence residual learning with BN
class EDSR_v108(BaseModel):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 3

  def extract_features(self, reuse=False):
    self.extract_recurrence_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

# for testing EDSRStepResidualTradeoff
class EDSRStepResidualTradeoff(BaseModel):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 2, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 2

  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

class EDSR_v201(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v202(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v203(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 3, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v204(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 4, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v205(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 5, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 5
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v206(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 6, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v207(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 7, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 7
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v208(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 8, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v209(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 9, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 9
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v210(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 10, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 10
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v211(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v212(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v213(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v214(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v215(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 5, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 5
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v216(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 6, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v217(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 7, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 7
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v218(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 8, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v219(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 9, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 9
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v220(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 10, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 10
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v221(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v222(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v223(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v224(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 4, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v225(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 5, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 5
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v226(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 6, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v227(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 7, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 7
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v228(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v229(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 9, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 9
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v230(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 10, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 10
    self.kernel_size = 3
    self.residual_depth = 2


class EDSR_v241(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 1


class EDSR_v242(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v243(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 3

class EDSR_v244(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x4
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 4

class EDSR_v245(EDSRStepResidualTradeoff):
  '''
    upscale: 2, step_depth: 2, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 5

class EDSR_v246(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 1

class EDSR_v247(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v248(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 3

class EDSR_v249(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x4
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 4

class EDSR_v250(EDSRStepResidualTradeoff):
  '''
    upscale: 4, step_depth: 4, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5

class EDSR_v251(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 1

class EDSR_v252(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 2

class EDSR_v253(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 3

class EDSR_v254(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x4
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 4

class EDSR_v255(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 8, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5

class EDSR_v256(EDSRStepResidualTradeoff):
  '''
    upscale: 8, step_depth: 6, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 6
    self.kernel_size = 3
    self.residual_depth = 5

# for test expand-squeeze block
class ExpandSqueezeBaseModel(BaseModel):
  def extract_features(self, reuse=False):
    self.extract_ed_block_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

  def total_variation_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      # loss = loss + total_variation_loss(self.sr_imgs[l])
      loss = loss + tf.reduce_sum(tf.image.total_variation(self.sr_imgs[l]))

    return loss

class EDSR_v301(ExpandSqueezeBaseModel):
  '''
    image_tune: 64x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 64
    self.image_g_kernel_size = 3

class EDSR_v302(ExpandSqueezeBaseModel):
  '''
    image_tune: 128x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 128
    self.image_g_kernel_size = 3

class EDSR_v303(ExpandSqueezeBaseModel):
  '''
    image_tune: 256x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 256
    self.image_g_kernel_size = 3

class EDSR_v304(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_v305(ExpandSqueezeBaseModel):
  '''
    image_tune: 1024x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 1024
    self.image_g_kernel_size = 3

class EDSR_v306(ExpandSqueezeBaseModel):
  '''
    image_tune: 64x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 64
    self.image_g_kernel_size = 5

class EDSR_v307(ExpandSqueezeBaseModel):
  '''
    image_tune: 128x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 128
    self.image_g_kernel_size = 5

class EDSR_v308(ExpandSqueezeBaseModel):
  '''
    image_tune: 256x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 256
    self.image_g_kernel_size = 5

class EDSR_v309(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_v310(ExpandSqueezeBaseModel):
  '''
    image_tune: 1024x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 1024
    self.image_g_kernel_size = 5

class EDSR_v311(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 1

class EDSR_v312(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_v313(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_v314(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 7
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 7

class EDSR_v315(ExpandSqueezeBaseModel):
  '''
    image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 9
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 1


class EDSR_v321(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 64x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 64
    self.image_g_kernel_size = 3

class EDSR_v322(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 128x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 128
    self.image_g_kernel_size = 3

class EDSR_v323(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 256x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 256
    self.image_g_kernel_size = 3

class EDSR_v324(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_v325(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 1024x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 1024
    self.image_g_kernel_size = 3

class EDSR_v326(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 1
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 1

class EDSR_v327(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_v328(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_v329(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 7
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 7

class EDSR_v330(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 8, residual_depth: 5x2, image_g_kernel_size: 9
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 9


# weighted loss


# class EDSR_v250(EDSRStepResidualTradeoff):
#   '''
#     upscale: 4, step_depth: 4, residual_depth: 2x5
#   '''
#   def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

#     BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

#     self.step_depth = 4
#     self.kernel_size = 3
#     self.residual_depth = 5
# for testing EDSRStepResidualTradeoff
class EDSRWeightLoss(BaseModel):
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5

  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)

  def l1_loss(self):
    return self.weighted_loss()

class EDSR_V500(EDSRWeightLoss):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 4, step_depth: 4, residual_depth: 2x5
  '''
  pass

class EDSR_V501(EDSRWeightLoss):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 2, step_depth: 2, residual_depth: 2x6
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 6

class EDSR_V502(EDSRWeightLoss):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 6, step_depth: 2, residual_depth: 2x4
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 4


class EDSRWeightLossES(ExpandSqueezeBaseModel):
  '''
    image_tune: 64x1 step_depth: 2, residual_depth: 2x6, image_g_kernel_size: 3
    contrast for expand-squeeze block
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 6
    self.image_squeeze_channle = 64
    self.image_g_kernel_size = 3

  def l1_loss(self):
    return self.weighted_loss()

class EDSR_V510(EDSRWeightLossES):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 6, step_depth: 2, residual_depth: 2x6
  '''
  pass

class EDSR_V511(EDSRWeightLossES):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 6, step_depth: 4, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 64
    self.image_g_kernel_size = 3

# for SR LFW Faces
class EDSR_LFW_v1(ExpandSqueezeBaseModel):
  '''
    upscale_factor=2 image_tune: 512x1 step_depth: 2, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_LFW_v2(ExpandSqueezeBaseModel):
  '''
    upscale_factor=2 image_tune: 512x1 step_depth: 2, residual_depth: 5x2, image_g_kernel_size: 5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_LFW_v3(ExpandSqueezeBaseModel):
  '''
    upscale_factor=4 image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class EDSR_LFW_v4(ExpandSqueezeBaseModel):
  '''
    upscale_factor=4 image_tune: 512x1 step_depth: 4, residual_depth: 5x2, image_g_kernel_size: 5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 4
    self.kernel_size = 3
    self.residual_depth = 5
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_LFW_v5(ExpandSqueezeBaseModel):
  '''
    upscale_factor=8 image_tune: 512x1 step_depth: 4, residual_depth: 10x2, image_g_kernel_size: 5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=8, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 8
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 5

class EDSR_LFW_v6(ExpandSqueezeBaseModel):
  '''
    upscale_factor=4 image_tune: 512x1 step_depth: 4, residual_depth: 10x2, image_g_kernel_size: 3
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    ExpandSqueezeBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 10
    self.image_squeeze_channle = 512
    self.image_g_kernel_size = 3

class LapSRNBaseModel(BaseModel):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 2, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

  def extract_features(self, reuse=False):
    self.extract_recurrence_features_without_BN(reuse)

  def reconstruct(self, reuse=False):
    self.residual_reconstruct(reuse)


class LapSRN_baseline_x2(LapSRNBaseModel):
  '''
    upscale: 2, step_depth: 1, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    LapSRNBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 5

class LapSRN_baseline_x4(LapSRNBaseModel):
  '''
    upscale: 4, step_depth: 2, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    LapSRNBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 5

class LapSRN_baseline_x8(LapSRNBaseModel):
  '''
    upscale: 8, step_depth: 3, residual_depth: 2x5
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=4, filter_num=64, reg=5e-4, scope='edsr'):

    LapSRNBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 3
    self.kernel_size = 3
    self.residual_depth = 5


class SRGANBaseModel(BaseModel):
  '''
    for step-num and residual-depth trade-off test, init params are
    upscale: 2, step_depth: 1, residual_depth: 2x2
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=1e-4, scope='edsr'):

    BaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 2
    self.kernel_size = 3
    self.residual_depth = 2

  def extract_features(self, reuse=False):
    self.extract_ed_block_features_with_BN(reuse)

  def reconstruct(self, reuse=False):
    self.normal_reconstruct(reuse)

class SRGAN_x2(SRGANBaseModel):
  '''
    upscale: 2, step_depth: 1, residual_depth: 2x12
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    SRGANBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 14

class SRGAN_x4(SRGANBaseModel):
  '''
    upscale: 4, step_depth: 1, residual_depth: 2x16
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    SRGANBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 16

class SRGAN_x8(SRGANBaseModel):
  '''
    upscale: 8, step_depth: 1, residual_depth: 2x16
  '''
  def __init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor=2, filter_num=64, reg=5e-4, scope='edsr'):

    SRGANBaseModel.__init__(self, inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size, is_training, upscale_factor, filter_num, reg, scope)

    self.step_depth = 1
    self.kernel_size = 3
    self.residual_depth = 18
