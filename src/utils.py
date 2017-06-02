from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random

def process_train_img(img, size, scale):
  with tf.variable_scope('process_img'):
    batch_size = tf.Tensor.get_shape(img).as_list()[0]
    height, width, channel = size

    rd = random.uniform(0.5, 1.2)
    random_height, random_width = int(height*rd), int(height*rd)
    gt_img = tf.random_crop(img, [batch_size, random_height, random_width, channel], seed=random.randint(1, 100))
    gt_img = tf.image.resize_bicubic(gt_img, [height, width], align_corners=False)

    # flip_lr_img = tf.split(value=gt_img, num_or_size_splits=batch_size, axis=0)
    # gt_img = tf.concat([tf.expand_dims(tf.image.random_flip_left_right(f_img[0]), axis=0) for f_img in flip_lr_img], axis=0)
    # flip_ud_img = tf.split(value=gt_img, num_or_size_splits=batch_size, axis=0)
    # gt_img = tf.concat([tf.expand_dims(tf.image.random_flip_up_down(f_img[0]), axis=0) for f_img in flip_ud_img], axis=0)

    input_img = tf.image.resize_bicubic(gt_img, [height//scale, width//scale], align_corners=False)

    # return gt_img, input_img
    return trainsform(input_img), trainsform(gt_img)

def trainsform(img):
  trains_img = tf.divide(img, 255.0)

  return trains_img

def transform_reverse(image):
  upscale_img = tf.multiply(image, 255.0)
  upscale_img = tf.clip_by_value(upscale_img, 0, 255)
  # upscale_img = tf.divide(tf.add(image, 1.0), 2.0)
  # upscale_img = tf.cast(tf.multiply(tf.add(image, 1.0), 127.5), tf.uint8)
  # upscale_img = tf.cast(tf.multiply(tf.add(image, 1.0), 127.5), tf.uint8)

  return upscale_img

def mse(target, base):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, base))))

def tf_flag_setup(flags):

  flags.DEFINE_string('dataset_dir', './dataset/train.h5', "the dataset path")
  flags.DEFINE_integer('gpu_id', 0, "max epoch to train")
  flags.DEFINE_integer('epoches', 5, "max epoch to train")
  flags.DEFINE_integer('batch_size', 10, "batch size")

  flags.DEFINE_string('g_ckpt_dir', './ckpt/lapsrn', "folder to save generator train models")

  flags.DEFINE_string('g_log_dir', './log/lapsrn-g', "folder to log generator train models")

  flags.DEFINE_integer('g_decay_steps', 100, "decay learning rate by intervel steps")
  flags.DEFINE_float('g_decay_rate', 0.9, "decay rate by intervel steps")
  flags.DEFINE_float('lr', 1e-4, "g learning rate")

  flags.DEFINE_integer('upscale_factor', 4, "upscale factor")

  flags.DEFINE_bool('is_training_mode', True, "whether continued training from last ckpt")
  flags.DEFINE_bool('continued_training', False, "whether continued training from last ckpt")
  flags.DEFINE_bool('debug', False, "whether or not to print debug messages")

  # Sets the graph-level random seed.
  tf.set_random_seed(random.randint(1, 10000))

def setup_project(FLAGS):
  # init dirs

  if tf.gfile.Exists(FLAGS.dataset_dir) == False:
    tf.gfile.MakeDirs(FLAGS.dataset_dir)

  if tf.gfile.Exists('./log') == False:
    tf.gfile.MakeDirs('./log')

  if tf.gfile.Exists('./ckpt') == False:
    tf.gfile.MakeDirs('./ckpt')

  if tf.gfile.Exists(FLAGS.g_ckpt_dir) == False:
    tf.gfile.MakeDirs(FLAGS.g_ckpt_dir)

def sess_configure(log_device_placement=False, memory_per=0.95):
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = memory_per

  return config
