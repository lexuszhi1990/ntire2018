import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from src.ops import *

class LapSRN(object):
  def __init__(self, upscale_factor=4, filter_num=64, is_training=True):
    self.scope = 'lap_srn'
    self.upscale_factor = upscale_factor
    self.filter_num = filter_num
    self.residual_depth = 10
    self.kernel_size = 3
    self.level = np.log2(upscale_factor).astype(int)
    self.sr_imgs = {}
    self.width = 64
    self.height = 64

  def __call__(self, inputs, reuse=False):
    with tf.variable_scope(self.scope) as vs:
      if reuse:
        vs.reuse_variables()

      net = layers.conv2d(inputs, self.filter_num, kernel_size=self.kernel_size, padding='SAME', activation_fn=lrelu, scope='init')

      for l in range(self.level):

        for d in range(self.residual_depth)
          net = layers.conv2d(net, self.filter_num, kernel_size=self.kernel_size, scope='level_{}_residual_{}'.format(str(l), str(d)))
        net = layers.conv2d_transpose(net, filter_num, kernel_size=kernel_size, stride=2, padding='VALID', activation_fn=lrelu, scope='level_{}_transpose'.format(str(level)))
        net = net[:,0:-1, 0:-1, :]
        net = layers.conv2d(net, 3, kernel_size=self.kernel_size, padding='SAME', activation_fn=None, scope='level_{}_features'.format(str(level)))

        bilinear_img = tf.image.resize_bilinear(inputs, size=[self.height*np.exp2(l+1).astype(int), self.width*np.exp2(l+1).astype(int)], align_corners=True, name='level_{}_biliear'.format(str(l)))

        net = tf.nn.tanh(bilinear_img + net)
        self.sr_imgs['level_{}_sr_img'.format(str(l))] = net

      return net

  @property
  def vars(self):
      return [var for var in tf.trainable_variables() if self.scope in var.name]
