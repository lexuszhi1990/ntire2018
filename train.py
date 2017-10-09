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

from src.model_new import EDSR_v100, EDSR_v101, EDSR_v102, EDSR_v103, EDSR_v104, EDSR_v105
from src.model_new import EDSR_v200, EDSR_v201, EDSR_v202, EDSR_v203, EDSR_v204, EDSR_v205, EDSR_v206, EDSR_v207, EDSR_v208, EDSR_v209, EDSR_v210, EDSR_v211, EDSR_v212, EDSR_v213, EDSR_v214, EDSR_v215, EDSR_v216, EDSR_v217, EDSR_v218, EDSR_v219, EDSR_v220, EDSR_v221, EDSR_v222, EDSR_v223, EDSR_v224, EDSR_v225, EDSR_v226, EDSR_v227, EDSR_v228, EDSR_v229, EDSR_v230, EDSR_v241, EDSR_v242, EDSR_v243, EDSR_v244, EDSR_v245, EDSR_v246, EDSR_v247, EDSR_v248, EDSR_v249, EDSR_v250, EDSR_v251, EDSR_v252, EDSR_v253, EDSR_v254, EDSR_v255
# from sr.model_new import EDSR_v231, EDSR_v232, EDSR_v233, EDSR_v234, EDSR_v235, EDSR_v236, EDSR_v237, EDSR_v238, EDSR_v239, EDSR_v240
from src.model_new import EDSR_v301

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project, sess_configure, tf_flag_setup, transform_reverse

def train(batch_size, upscale_factor, epoches, lr, reg, filter_num, g_decay_rate, g_decay_steps, dataset_dir, g_ckpt_dir, g_log_dir, gpu_id, continued_training, model_name, model_path, debug):

  model_list = []
  sess_conf = sess_configure()
  graph = tf.Graph()

  dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(gpu_id))):

      batch_gt_x2 = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      batch_gt_x4 = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      batch_gt_x8 = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      batch_inputs = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      is_training = tf.placeholder(tf.bool, [])

      SRNet = globals()[model_name]
      model = SRNet(batch_inputs, batch_gt_x2, batch_gt_x4, batch_gt_x8, image_size=dataset.input_size, is_training=is_training, upscale_factor=dataset.upscale, reg=reg, filter_num=filter_num)
      model.init_gt_imgs()
      model.extract_features()
      model.reconstruct()
      loss = model.l1_loss()

      upscaled_x2_img = transform_reverse(model.upscaled_img(2))
      upscaled_x4_img = transform_reverse(model.upscaled_img(4))
      batch_input_sum = tf.summary.image("inputs", transform_reverse(batch_inputs), max_outputs=2)
      gt_bicubic_sum = tf.summary.image("bicubic_img", transform_reverse(tf.image.resize_images(batch_inputs, size=[dataset.gt_height, dataset.gt_width], method=tf.image.ResizeMethod.BICUBIC, align_corners=False)), max_outputs=2)
      gt_sum = tf.summary.image("gt", transform_reverse(batch_gt_x4), max_outputs=2)
      g_output_sum = tf.summary.image("upscaled", upscaled_x4_img, max_outputs=2)
      g_loss_sum = tf.summary.scalar("g_loss", loss)

      counter = tf.get_variable(name="counter", shape=[], initializer=tf.constant_initializer(0), trainable=False)
      lr = tf.train.exponential_decay(lr, counter, decay_rate=g_decay_rate, decay_steps=g_decay_steps, staircase=True)
      # opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, momentum=0.9, epsilon=1e-8)
      opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
      grads = opt.compute_gradients(loss, var_list=model.vars)
      apply_gradient_opt = opt.apply_gradients(grads, global_step=counter)
      g_lr_sum = tf.summary.scalar("g_lr", lr)

      # restore model
      all_variables = tf.global_variables()
      saver = tf.train.Saver(all_variables, max_to_keep=10)
      if continued_training:
        saver.restore(sess, model_path)
        print('restore the g from %s'%model_path)
      else:
        print('there is no ckpt for g...')
        sess.run(tf.variables_initializer(set(all_variables)))

      # rebuild log dir
      summary_writer = tf.summary.FileWriter(g_log_dir, sess.graph)

      g_sum_all = tf.summary.merge([g_output_sum, gt_sum, gt_bicubic_sum, batch_input_sum, g_loss_sum, g_lr_sum])

      for epoch in range(1, epoches+1):
        for step in range(1, dataset.batch_ids+1):

          batch_img_x8, batch_img_x4, batch_img_x2, batch_in = dataset.next_batch(step-1)

          # if step % (dataset.batch_ids//3) == 0:
          if step % (dataset.batch_ids//500) == 0:
            merged, apply_gradient_opt_, lr_, loss_ = sess.run([g_sum_all, apply_gradient_opt, lr, loss], feed_dict={batch_gt_x2: batch_img_x2, batch_gt_x4: batch_img_x4, batch_gt_x8: batch_img_x8, batch_inputs: batch_in, is_training: True})
            print("at %d/%d, lr_: %.5f, g_loss: %.5f" % (epoch, step, lr_, loss_))
            summary_writer.add_summary(merged, step + epoch*dataset.batch_ids)
          else:
            apply_gradient_opt_, lr_, loss_ = sess.run([apply_gradient_opt, lr, loss], feed_dict={batch_gt_x2: batch_img_x2, batch_gt_x4: batch_img_x4, batch_gt_x8: batch_img_x8, batch_inputs: batch_in, is_training: True})
            print("at %d/%d, lr_: %.5f, g_loss: %.5f" % (epoch, step, lr_, loss_))

        # if epoch % (epoches//2) == 0:
        if epoch == epoches:
          ckpt_name = "{}-epoch-{}-step-{}-{}.ckpt".format(model_name, epoch, step, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
          saver.save(sess, os.path.join(g_ckpt_dir, ckpt_name), global_step=step)
          model_list.append(os.path.join(g_ckpt_dir, "{}-{}".format(ckpt_name, step)))
          print('save model at step: %d, in dir %s, name %s' %(step, g_ckpt_dir, ckpt_name))

      return model_list[-1]
