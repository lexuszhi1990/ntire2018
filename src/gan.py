import numpy as np
import tensorflow as tf

from src.layer import *
import tensorflow.contrib.layers as layers

from src.model_new import ExpandSqueezeBaseModel
from src.model_new import EDSR_LFW_v1, EDSR_LFW_v2, EDSR_LFW_v3, EDSR_LFW_v4, EDSR_LFW_v5

# import sys
# sys.path.append('../squeeze_det/lib')
# sys.path.append('../squeeze_det/config')
# from squeeze_det.config import *
# from squeeze_det.lib import ResNet50ConvDetForward, VGG16ConvDetForward, SqueezeDetPlusForward

class EDSR_v401(ExpandSqueezeBaseModel):
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

  def total_variation_loss(self):
    loss = 0.0
    for l in range(self.step_depth):
      # loss = loss + total_variation_loss(self.sr_imgs[l])
      loss = loss + tf.reduce_sum(tf.image.total_variation(self.sr_imgs[l]))

    return loss

class Discriminator(object):
  def __init__(self, is_training=True, filter_num=64):
    self.scope = 'discriminator'
    self.filter_num = filter_num

  @property
  def vars(self):
      return [var for var in tf.trainable_variables() if self.scope in var.name]

  def __call__(self, inputs, reuse=None):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      batch_size = tf.Tensor.get_shape(inputs).as_list()[0]

      d_h0_1 = layers.conv2d(inputs, self.filter_num, [3, 3], normalizer_fn = layers.batch_norm, stride=1, activation_fn=lrelu, scope='d_h0_1')
      d_h0_2 = layers.conv2d(d_h0_1, self.filter_num, [4, 4], normalizer_fn = layers.batch_norm, stride=2, activation_fn=lrelu, scope='d_h0_2')
      d_h1_1 = layers.conv2d(d_h0_2, self.filter_num * 1, [3, 3], normalizer_fn = layers.batch_norm, stride=1, activation_fn=lrelu, scope='d_h1_1')
      d_h1_2 = layers.conv2d(d_h1_1, self.filter_num * 1, [4, 4], normalizer_fn = layers.batch_norm, stride=2, activation_fn=lrelu, scope='d_h1_2')
      d_h2_1 = layers.conv2d(d_h1_2, self.filter_num * 2, [3, 3], normalizer_fn = layers.batch_norm, stride=1, activation_fn=lrelu, scope='d_h2_1')
      d_h2_2 = layers.conv2d(d_h2_1, self.filter_num * 2, [4, 4], normalizer_fn = layers.batch_norm, stride=2, activation_fn=lrelu, scope='d_h2_2')
      d_h3_1 = layers.conv2d(d_h2_2, self.filter_num * 4, [3, 3], normalizer_fn = layers.batch_norm, stride=1, activation_fn=lrelu, scope='d_h3_1')
      d_h3_2 = layers.conv2d(d_h3_1, self.filter_num * 4, [4, 4], normalizer_fn = layers.batch_norm, stride=2, activation_fn=lrelu, scope='d_h3_2')
      logit_1024 = layers.fully_connected(tf.reshape(
          d_h3_2, [batch_size, -1]), 512, activation_fn=lrelu, weights_regularizer=layers.l2_regularizer(0.05), scope='fc1')
      logit = layers.fully_connected(logit_1024, 1, activation_fn=None, scope='fc2')

      return tf.nn.sigmoid(logit), logit

class SRGAN(object):
  def __init__(self, model_name, batch_size, upscale, channel, inputs_size, g_decay_steps, d_decay_steps, d_filter_num, g_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate, is_train, scope='SRGAN'):
    self.scope = scope

    self.model_name = model_name
    self.is_training = is_train
    self.batch_size, self.upscale, self.channel = batch_size, upscale, channel
    self.in_height, self.in_width = inputs_size
    self.gt_height, self.gt_width = self.in_height * self.upscale, self.in_width * self.upscale

    self.g_decay_steps = g_decay_steps
    self.d_decay_steps = d_decay_steps

    self.d_filter_num = d_filter_num
    self.g_filter_num = g_filter_num
    self.g_init_lr = g_lr
    self.d_init_lr = d_lr
    self.g_decay_rate = g_decay_rate
    self.d_decay_rate = d_decay_rate

  @property
  def vars(self):
    return [var for var in tf.trainable_variables() if self.scope in var.name]

  def init_placeholders(self):
    self.is_training = tf.placeholder(tf.bool, [])
    self.batch_inputs = tf.placeholder(tf.float32, [self.batch_size, None, None, self.channel])
    for scale in [2, 4, 8]:
      # vars(self)["batch_gt_x{}".format(scale)] = tf.placeholder(tf.float32, [self.batch_size, self.in_height * scale, self.in_width * scale, self.channel])
      vars(self)["batch_gt_x{}".format(scale)] = tf.placeholder(tf.float32, [self.batch_size, None, None, self.channel])
    self.batch_gt_img = vars(self)["batch_gt_x{}".format(self.upscale)]

    self.counter_g = tf.get_variable(name="counter_g", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    self.counter_d = tf.get_variable(name="counter_d", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    self.clamp_lower, self.clamp_upper = [0.01, 0.01]

  def build_g_graph(self):

    g_model = globals()[self.model_name]
    self.generator = g_model(self.batch_inputs, self.batch_gt_x2, self.batch_gt_x4, self.batch_gt_x8, image_size=[self.in_height, self.in_width], is_training=self.is_training, upscale_factor=self.upscale, filter_num=self.g_filter_num, reg=1e-4)

    self.generator.init_gt_imgs()
    self.generator.extract_features()
    self.generator.reconstruct()
    self.g_result = self.generator.upscaled_img(self.upscale)

  def calculate_gan_d_loss(self):
    self.discriminator = Discriminator(filter_num=self.d_filter_num)
    D_real, D_real_logit = self.discriminator(self.batch_gt_img)
    D_fake, D_fake_logit = self.discriminator(self.g_result, reuse=True)

    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake)))
    self.g_loss_pred = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake)))
    self.d_loss = self.d_loss_real + self.d_loss_fake

  def calculate_wgan_d_loss(self):
    self.discriminator = Discriminator(filter_num=self.d_filter_num)
    D_real, D_real_logit = self.discriminator(self.batch_gt_img)
    D_fake, D_fake_logit = self.discriminator(self.g_result, reuse=True)

    self.d_loss_real = tf.reduce_mean(D_real_logit)
    self.d_loss_fake = tf.reduce_mean(D_fake_logit)

    self.d_loss = tf.reduce_mean(D_fake_logit - D_real_logit)
    self.g_loss_pred = tf.reduce_mean(-D_fake_logit)
    # self.d_loss_real = tf.reduce_mean(tf.scalar_mul(-1, D_real_logit))
    # self.d_loss_fake = tf.reduce_mean(D_fake_logit)
    # self.d_loss = self.d_loss_real + self.d_loss_fake

  def calculate_g_loss(self):
    self.g_context_loss = self.generator.l1_loss()
    self.g_perceptual_loss = self.generator.l1_loss()
    self.g_tv_loss = self.generator.total_variation_loss()

    # self.g_loss = self.g_context_loss + 1E-3 * self.g_tv_loss + 1E-5 * self.g_loss_pred
    self.g_loss = self.g_context_loss + 1E-5 * self.g_loss_pred

  def gan_backward(self):
    self.g_lr = tf.train.exponential_decay(self.g_init_lr, self.counter_g, decay_rate=self.g_decay_rate, decay_steps=self.g_decay_steps, staircase=True)
    g_opt = tf.train.RMSPropOptimizer(learning_rate=self.g_lr, decay=0.95, momentum=0.9, epsilon=1e-8)
    g_grads = g_opt.compute_gradients(self.g_loss, var_list=self.generator.vars)
    self.g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=self.counter_g)

    self.d_lr = tf.train.exponential_decay(self.d_init_lr, self.counter_d, decay_rate=self.d_decay_rate, decay_steps=self.d_decay_steps, staircase=True)
    d_opt = tf.train.RMSPropOptimizer(learning_rate=self.d_lr, decay=0.95, momentum=0.9, epsilon=1e-8)
    d_grads = d_opt.compute_gradients(self.d_loss, var_list=self.discriminator.vars)
    self.d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=self.counter_d)

  def wgan_backward_append(self):
    clipped_var_d = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in self.discriminator.vars]
    # merge the clip operations on critic variables
    with tf.control_dependencies([self.d_apply_gradient_op]):
      self.d_apply_gradient_op = tf.tuple(clipped_var_d)

  def build_summary(self):

    d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    g_loss_pred_sum = tf.summary.scalar("g_loss_pred", self.g_loss_pred)
    d_lr_sum = tf.summary.scalar("d_lr", self.d_lr)
    d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    g_output_sum = tf.summary.image("upscaled", self.g_result, max_outputs=2)
    gt_output_sum = tf.summary.image("gt", self.batch_gt_img, max_outputs=2)
    batch_input_img_sum = tf.summary.image("inputs", self.batch_inputs, max_outputs=2)
    bicubic_output_sum = tf.summary.image("bicubic_upsaced", tf.image.resize_images(self.batch_inputs, size=[self.gt_height, self.gt_width], method=tf.image.ResizeMethod.BICUBIC), max_outputs=2)
    g_tv_loss_sum = tf.summary.scalar("g_tv_loss", self.g_tv_loss)
    g_context_loss_sum = tf.summary.scalar("g_context_loss", self.g_context_loss)
    g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    g_lr_sum = tf.summary.scalar("g_lr", self.g_lr)

    self.d_sum_all = tf.summary.merge([g_loss_pred_sum, d_loss_sum, d_lr_sum, d_loss_real_sum, d_loss_fake_sum])
    self.g_sum_all = tf.summary.merge([g_output_sum, gt_output_sum, batch_input_img_sum, bicubic_output_sum, g_tv_loss_sum, g_context_loss_sum, g_loss_sum, g_lr_sum])

  def d_operations(self):
    return [self.d_apply_gradient_op, self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_lr]

  def d_operations_with_summary(self):
    return [self.d_sum_all] + self.d_operations()

  def g_operations(self):
    return [self.g_apply_gradient_op, self.g_loss, self.g_context_loss, self.g_loss_pred, self.g_tv_loss, self.g_lr]

  def g_operations_with_summary(self):
    return [self.g_sum_all] + self.g_operations()

  def next_feed_dict(self, next_batch):
    batch_label_x8, batch_label_x4, batch_label_x2, batch_data = next_batch
    feed_dict = {self.batch_inputs: batch_data, self.batch_gt_x2: batch_label_x2, self.batch_gt_x4: batch_label_x4, self.batch_gt_x8: batch_label_x8, self.is_training: True}

    return feed_dict

  def set_retsored_variables(self):
    self.gr_vars = [var for var in tf.global_variables() if self.generator.scope in var.name]
    self.dr_vars = [var for var in tf.global_variables() if self.discriminator.scope in var.name]

  def build_wgan_model(self):
    self.init_placeholders()
    self.build_g_graph()
    self.calculate_wgan_d_loss()
    self.calculate_g_loss()
    self.gan_backward()
    self.wgan_backward_append()
    self.build_summary()

  def build_gan_model(self):
    self.init_placeholders()
    self.build_g_graph()
    self.calculate_gan_d_loss()
    self.calculate_g_loss()
    self.gan_backward()
    self.build_summary()

