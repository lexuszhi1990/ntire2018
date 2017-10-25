#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import numpy as np
import tensorflow as tf

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19
from src.model import LapSRN_v2_v1, LapSRN_v2_v2
from src.model import LapSRN_v30, LapSRN_v31, LapSRN_v32, LapSRN_v33, LapSRN_v34
from src.model import LapSRN_v40, LapSRN_v41, LapSRN_v42, LapSRN_v43, LapSRN_v44

from src.model_new import EDSR_v100, EDSR_v101, EDSR_v102, EDSR_v103, EDSR_v104, EDSR_v105, EDSR_v106, EDSR_v107, EDSR_v108
from src.model_new import EDSR_v201, EDSR_v202, EDSR_v203, EDSR_v204, EDSR_v205, EDSR_v206, EDSR_v207, EDSR_v208, EDSR_v209, EDSR_v210, EDSR_v211, EDSR_v212, EDSR_v213, EDSR_v214, EDSR_v215, EDSR_v216, EDSR_v217, EDSR_v218, EDSR_v219, EDSR_v220, EDSR_v221, EDSR_v222, EDSR_v223, EDSR_v224, EDSR_v225, EDSR_v226, EDSR_v227, EDSR_v228, EDSR_v229, EDSR_v230, EDSR_v241, EDSR_v242, EDSR_v243, EDSR_v244, EDSR_v245, EDSR_v246, EDSR_v247, EDSR_v248, EDSR_v249, EDSR_v250, EDSR_v251, EDSR_v252, EDSR_v253, EDSR_v254, EDSR_v255
# from sr.model_new import EDSR_v231, EDSR_v232, EDSR_v233, EDSR_v234, EDSR_v235, EDSR_v236, EDSR_v237, EDSR_v238, EDSR_v239, EDSR_v240
from src.model_new import EDSR_v301, EDSR_v302, EDSR_v303, EDSR_v304, EDSR_v305, EDSR_v306, EDSR_v307, EDSR_v308, EDSR_v309, EDSR_v310, EDSR_v311, EDSR_v312, EDSR_v313, EDSR_v314, EDSR_v315, EDSR_v321, EDSR_v322, EDSR_v323, EDSR_v324, EDSR_v325, EDSR_v326, EDSR_v327, EDSR_v328, EDSR_v329, EDSR_v330
from src.model_new import EDSR_LFW_v1, EDSR_LFW_v2, EDSR_LFW_v3, EDSR_LFW_v4

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project, sess_configure, tf_flag_setup, transform_reverse

# def train(batch_size, upscale_factor, epoches, lr, reg, filter_num, g_decay_rate, g_decay_steps, dataset_dir, ckpt_dir, g_log_dir, gpu_id, continued_training, model_name, debug):

# def train(batch_size, upscale_factor, inner_epoches, g_lr, reg, g_filter_num, d_filter_num, g_decay_rate, g_decay_steps, d_decay_rate, d_decay_steps, dataset_dir, ckpt_dir, log_dir, gpu_id, default_sr_method):

def train(log_dir, gpu_id, ckpt_dir, dataset_dir, model_name, batch_size, upscale, channel, g_decay_steps, d_decay_steps, g_filter_num, d_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate, is_wgan):

  model_list = []
  sess_conf = sess_configure()
  graph = tf.Graph()
  dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale)

  import pdb
  pdb.set_trace()

  with graph.as_default(), tf.Session(config=sess_configure()) as sess:
    with tf.device('/gpu:%d' % gpu_id ):
      SRNet = globals()[model_name]
      srgan = SRNet(batch_size, upscale, channel, dataset.inputs_size, g_decay_steps, d_decay_steps, d_filter_num, g_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate)

      with tf.variable_scope(srgan.scope):
        if is_wgan:
          srgan.build_wgan_model()
        else:
          srgan.build_gan_model()

    all_variables = set([ var for var in tf.global_variables() if srgan.scope in var.name])
    saver = tf.train.Saver(all_variables, max_to_keep=3)
    # gan_variables = set(srgan.vars)
    # gan_saver = tf.train.Saver(all_variables, max_to_keep=5)
    # if continued_training:
      # saver.restore(sess, model_path)
      # print("restore full model from %s"%model_path)
      # all_variables = all_variables - gan_variables
    sess.run(tf.variables_initializer(all_variables))
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    for step in range(1, dataset.batch_ids+1, 2):
      if step % (dataset.batch_ids//500) == 0:
        d_merged, _, d_loss_, d_loss_real_, d_loss_fake_, d_lr_ = sess.run(srgan.d_operations_with_summary(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step-1)))
        summary_writer.add_summary(d_merged, total_step)
      else:
        _, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_, g_lr_ = sess.run(srgan.g_operations(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step-1)))

      print('[epoch:%d][step:%d/%d], d_lr: %.6f, d_loss: %.6f, d_loss_real: %.6f, d_loss_fake: %.6f'%(epoch, step, j, d_lr_, d_loss_, d_loss_real_, d_loss_fake_))

      # train G
      if step % (dataset.batch_ids//500) == 1:
        g_merged, _, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_, g_lr_ = sess.run(srgan.g_operations_with_summary(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step)))
        summary_writer.add_summary(g_merged, total_step)
      else:
         _, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_, g_lr_ = sess.run(srgan.g_operations(), feed_dict=srgan.next_feed_dict(dataset.next_batch(step)))
      print("[epoch:%d][step:%d/g], g_lr: %.6f, g_loss: %.6f, g_ct_l: %.6f, g_pred_l: %.6f, g_tv_l: %.6f"%(epoch, step, g_lr_, g_loss_, g_context_loss_, g_loss_pred_, g_tv_loss_))


    ckpt_name = "{}-wgan-{}-epoch-{}-step-{}-{}.ckpt".format(model_name, is_wgan, step, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
    saver.save(sess, os.path.join(ckpt_dir, ckpt_name), global_step=step)
    model_list.append(os.path.join(ckpt_dir, "{}-{}".format(ckpt_name, step)))
    print('save model at step: %d, in dir %s, name %s' %(step, ckpt_dir, ckpt_name))

    return model_list[-1]
