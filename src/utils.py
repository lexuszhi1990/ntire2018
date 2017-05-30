from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def mse(target, base):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, base))))

def transform_reverse(image):
  upscale_img = tf.divide(tf.add(image, 1.0), 2.0)
  # upscale_img = tf.cast(tf.multiply(tf.add(image, 1.0), 127.5), tf.uint8)
  # upscale_img = tf.clip_by_value(tf.multiply(upscale_img, 255.0), 0, 255)
  # upscale_img = tf.cast(tf.multiply(tf.add(image, 1.0), 127.5), tf.uint8)

  return upscale_img

def tf_flag_setup(flags):

  flags.DEFINE_string('dataset_dir', './dataset/train.h5', "the dataset path")
  flags.DEFINE_integer('gpu_id', 0, "max epoch to train")
  flags.DEFINE_integer('epoches', 5, "max epoch to train")
  flags.DEFINE_integer('batch_size', 10, "batch size")

  flags.DEFINE_string('g_ckpt_dir', './ckpt/lapsrn', "folder to save generator train models")

  flags.DEFINE_string('g_log_dir', './log/lapsrn-g', "folder to log generator train models")

  flags.DEFINE_integer('g_decay_steps', 100, "decay learning rate by intervel steps")
  flags.DEFINE_float('g_decay_rate', 0.95, "decay rate by intervel steps")
  flags.DEFINE_float('lr', 0.00005, "g learning rate")

  flags.DEFINE_integer('upscale_factor', 4, "upscale factor")

  flags.DEFINE_bool('is_training_mode', True, "whether continued training from last ckpt")
  flags.DEFINE_bool('continued_training', False, "whether continued training from last ckpt")
  flags.DEFINE_bool('debug', False, "whether or not to print debug messages")

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
