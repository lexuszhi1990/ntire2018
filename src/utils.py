from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def mse(target, base):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, base))))

def transform_reverse(image):
  # return tf.multiply(tf.add(image, 1.0), 127.5)
  return tf.divide(tf.add(image, 1.0), 2.0)

def tf_flag_setup(flags):

  flags.DEFINE_string('dataset_dir', './dataset/train', "folder to save data")
  flags.DEFINE_string('dataset', 'set14', "folder to save data")
  flags.DEFINE_integer('gpu_id', 0, "max epoch to train")
  flags.DEFINE_integer('max_steps', 40000, "max epoch to train")
  flags.DEFINE_integer('batch_size', 10, "batch size")

  flags.DEFINE_string('g_ckpt_dir', './ckpt/lapsrn', "folder to save generator train models")

  flags.DEFINE_string('g_log_dir', './log/lapsrn-g', "folder to log generator train models")

  flags.DEFINE_integer('g_decay_steps', 5000, "decay learning rate by intervel steps")
  flags.DEFINE_float('g_decay_rate', 0.98, "decay rate by intervel steps")
  flags.DEFINE_float('lr', 0.0005, "g learning rate")

  flags.DEFINE_integer('upscale_factor', 4, "upscale factor")

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
