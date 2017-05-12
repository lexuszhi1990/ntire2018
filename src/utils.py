from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def tf_flag_setup(flags):

  flags.DEFINE_string('dataset_dir', './datasets/train', "folder to save data")
  flags.DEFINE_string('dataset', 'set14', "folder to save data")
  flags.DEFINE_integer('gpu_id', 0, "max epoch to train")
  flags.DEFINE_integer('max_steps', 40000, "max epoch to train")
  flags.DEFINE_integer('g_pretrain_steps', 10000, "pre-train g steps")
  flags.DEFINE_integer('d_pretrain_steps', 5000, "pre-train d steps")
  flags.DEFINE_integer('batch_size', 16, "batch size")

  flags.DEFINE_string('full_ckpt_dir', './ckpt/sr-gan-full', "folder to save full train models")
  flags.DEFINE_string('g_ckpt_dir', './ckpt/sr-gan-g', "folder to save generator train models")
  flags.DEFINE_string('d_ckpt_dir', './ckpt/sr-gan-d', "folder to save descriminator  models")

  flags.DEFINE_string('full_log_dir', './log/sr-gan', "folder to save full train models")
  flags.DEFINE_string('g_log_dir', './log/sr-gan-g', "folder to log generator train models")
  flags.DEFINE_string('d_log_dir', './log/sr-gan-d', "folder to log descriminator models")

  flags.DEFINE_integer('g_base_filter_num', 32, "d base filter num")
  flags.DEFINE_integer('d_base_filter_num', 32, "g base filter num")
  flags.DEFINE_integer('d_iters', 3, "disc iterations per generator iteration")
  flags.DEFINE_integer('g_decay_steps', 5000, "decay learning rate by intervel steps")
  flags.DEFINE_integer('d_decay_steps', 5000, "decay learning rate by intervel steps")
  flags.DEFINE_float('g_decay_rate', 0.98, "decay rate by intervel steps")
  flags.DEFINE_float('d_decay_rate', 0.98, "decay rate by intervel steps")
  flags.DEFINE_float('d_lr', 0.0005, "d learning rate")
  flags.DEFINE_float('g_lr', 0.0005, "g learning rate")
  flags.DEFINE_float('clamp_lower_d', -0.01, "lower clamp")
  flags.DEFINE_float('clamp_upper_d', 0.01, "upper clamp")

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

def sess_configure(log_device_placement=False, memory_per=0.95):
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = memory_per

  return config
