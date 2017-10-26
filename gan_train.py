#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import numpy as np
import tensorflow as tf

from src.gan import SRGAN, EDSR_v401

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project, sess_configure, tf_flag_setup, transform_reverse

# def train(batch_size, upscale_factor, epoches, lr, reg, filter_num, g_decay_rate, g_decay_steps, dataset_dir, ckpt_dir, g_log_dir, gpu_id, continued_training, model_name, debug):

# def train(batch_size, upscale_factor, inner_epoches, g_lr, reg, g_filter_num, d_filter_num, g_decay_rate, g_decay_steps, d_decay_rate, d_decay_steps, dataset_dir, ckpt_dir, log_dir, gpu_id, default_sr_method):

def train(log_dir, gpu_id, ckpt_dir, dataset_dir, model_name, batch_size, upscale, channel, g_decay_steps, d_decay_steps, g_filter_num, d_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate, is_wgan, is_train=False):

  model_list = []
  sess_conf = sess_configure()
  graph = tf.Graph()
  dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale)

  with graph.as_default(), tf.Session(config=sess_configure()) as sess:
    with tf.device('/gpu:%d' % gpu_id ):

      srgan = SRGAN(model_name, batch_size, upscale, channel, dataset.input_size, g_decay_steps, d_decay_steps, d_filter_num, g_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate, is_train)

      if is_wgan:
        srgan.build_wgan_model()
      else:
        srgan.build_gan_model()

    # all_variables = set([ var for var in tf.global_variables() if srgan.generator.scope in var.name or srgan.discriminator.scope in var.name ])
    all_variables = tf.global_variables()
    saver = tf.train.Saver(all_variables, max_to_keep=3)
    # gan_variables = set(srgan.vars)
    # gan_saver = tf.train.Saver(all_variables, max_to_keep=5)
    # if continued_training:
      # saver.restore(sess, model_path)
      # print("restore full model from %s"%model_path)
      # all_variables = all_variables - gan_variables
    sess.run(tf.variables_initializer(all_variables))
    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    for step in range(1, dataset.batch_ids+1, 2):
      if step % (dataset.batch_ids//500) == 0:
        d_merged, _, d_loss_, d_loss_real_, d_loss_fake_, d_lr_ = sess.run(srgan.d_operations_with_summary(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step-1)))
        summary_writer.add_summary(d_merged, step)
      else:
        _, d_loss_, d_loss_real_, d_loss_fake_, d_lr_ = sess.run(srgan.d_operations(), feed_dict=srgan.next_feed_dict(dataset.next_batch()))
      print('[step:%d][step:%d], d_lr: %.6f, d_loss: %.6f, d_loss_real: %.6f, d_loss_fake: %.6f'%(step, step, d_lr_, d_loss_, d_loss_real_, d_loss_fake_))

      # train G
      if step % (dataset.batch_ids//500) == 1:
        g_merged, _, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_, g_lr_ = sess.run(srgan.g_operations_with_summary(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step)))
        summary_writer.add_summary(g_merged, step)
      else:
         _, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_, g_lr_ = sess.run(srgan.g_operations(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step)))
      print("[step:%d][step:%d/g], g_lr: %.6f, g_loss: %.6f, g_ct_l: %.6f, g_pred_l: %.6f, g_tv_l: %.6f"%(step, step, g_lr_, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_))


    ckpt_name = "{}-wgan-{}-step-{}-{}.ckpt".format(model_name, is_wgan, step, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
    saver.save(sess, os.path.join(ckpt_dir, ckpt_name), global_step=step)
    model_list.append(os.path.join(ckpt_dir, "{}-{}".format(ckpt_name, step)))
    print('save model at step: %d, in dir %s, name %s' %(step, ckpt_dir, ckpt_name))

    return model_list[-1]
