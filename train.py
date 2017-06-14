#!/usr/bin/python
'''
usage:
  python train.py --dataset_dir=./dataset/train.h5 --continued_training=False --batch_size=8 --gpu_id=2 --epoches=100 --lr=0.0004
  python train.py --dataset_dir=./dataset/lap_pry_x4_small.h5 --continued_training=False --batch_size=8 --gpu_id=2 --epoches=100 --lr=0.0004
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import numpy as np
import tensorflow as tf

from src.model import LapSRN, LapSRN_v1
from src.dataset import TrainDataset, DatasetFromHdf5, DatasetFromHdf5V1
from src.utils import setup_project, sess_configure, tf_flag_setup, transform_reverse

# for log infos
pp = pprint.PrettyPrinter()
tf.logging.set_verbosity(tf.logging.INFO)
info = tf.logging.info

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS
tf_flag_setup(flags)

def train(options):
  # define training variables here
  batch_size = FLAGS.batch_size
  dataset_dir = FLAGS.dataset_dir
  g_ckpt_dir = FLAGS.g_ckpt_dir
  gpu_id = FLAGS.gpu_id
  g_decay_rate = FLAGS.g_decay_rate
  upscale_factor = FLAGS.upscale_factor
  continued_training = FLAGS.continued_training
  epoches = FLAGS.epoches
  g_log_dir = FLAGS.g_log_dir
  debug = FLAGS.debug

  lr = FLAGS.lr
  sess_conf = sess_configure()
  graph = tf.Graph()

  dataset = DatasetFromHdf5V1(file_path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)
  g_decay_steps = np.floor(np.log(g_decay_rate)/np.log(0.1) * (dataset.batch_ids*epoches))

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(gpu_id))):

      batch_gt_x2 = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      batch_gt_x4 = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      batch_inputs = tf.placeholder(tf.float32, [batch_size, None, None, dataset.channel])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN_v1(batch_inputs, batch_gt_x2, batch_gt_x4, image_size=dataset.inputs_size, is_training=is_training, upscale_factor=dataset.upscale)
      model.extract_features()
      model.reconstruct()
      loss = model.l1_loss()

      upscaled_x2_img = transform_reverse(model.sr_imgs[0])
      upscaled_x4_img = transform_reverse(model.sr_imgs[1])
      batch_input_sum = tf.summary.image("inputs", transform_reverse(batch_inputs), max_outputs=2)
      gt_bicubic_sum = tf.summary.image("bicubic_img", transform_reverse(tf.image.resize_images(batch_inputs, size=[dataset.gt_height, dataset.gt_width], method=tf.image.ResizeMethod.BICUBIC, align_corners=False)), max_outputs=2)
      gt_sum = tf.summary.image("gt", transform_reverse(batch_gt_x4), max_outputs=2)
      g_output_sum = tf.summary.image("upscaled", upscaled_x4_img, max_outputs=2)
      g_loss_sum = tf.summary.scalar("g_loss", loss)

      counter = tf.get_variable(name="counter", shape=[], initializer=tf.constant_initializer(0), trainable=False)
      lr = tf.train.exponential_decay(lr, counter, decay_rate=g_decay_rate, decay_steps=g_decay_steps, staircase=True)
      opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, momentum=0.9, epsilon=1e-8)
      # opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
      grads = opt.compute_gradients(loss, var_list=model.vars)
      apply_gradient_opt = opt.apply_gradients(grads, global_step=counter)
      g_lr_sum = tf.summary.scalar("g_lr", lr)

      # restore model
      all_variables = tf.global_variables()
      saver = tf.train.Saver(all_variables, max_to_keep=10)
      ckpt = tf.train.get_checkpoint_state(g_ckpt_dir)
      if ckpt and continued_training:
        saver.restore(sess, ckpt.model_checkpoint_path)
        info('restore the g from %s', ckpt.model_checkpoint_path)
        if debug:
          [print(v.name) for v in all_variables]
          print("all D variable" , len(all_variables))
          print("all global_variables" , len(tf.global_variables()))
          print("all local_variables" , len(tf.local_variables()))
      else:
        info('there is no ckpt for g...')
        sess.run(tf.variables_initializer(set(all_variables)))

      # rebuild log dir
      if tf.gfile.Exists(g_log_dir):
        tf.gfile.DeleteRecursively(g_log_dir)
      tf.gfile.MakeDirs(g_log_dir)
      summary_writer = tf.summary.FileWriter(g_log_dir, sess.graph)

      g_sum_all = tf.summary.merge([g_output_sum, gt_sum, gt_bicubic_sum, batch_input_sum, g_loss_sum, g_lr_sum])

      for epoch in range(1, epoches+1):
        for step in range(1, dataset.batch_ids+1):

          batch_img_x4, batch_img_x2, batch_in, _ = dataset.next(step-1)

          if step % (dataset.batch_ids//3) == 0:
            merged, apply_gradient_opt_, lr_, loss_ = sess.run([g_sum_all, apply_gradient_opt, lr, loss], feed_dict={batch_gt_x2: batch_img_x2, batch_gt_x4: batch_img_x4, batch_inputs: batch_in, is_training: True})
            info("at %d/%d, lr_: %.5f, g_loss: %.5f", epoch, step, lr_, loss_)
            summary_writer.add_summary(merged, step + epoch*dataset.batch_ids)
          else:
            apply_gradient_opt_, lr_, loss_ = sess.run([apply_gradient_opt, lr, loss], feed_dict={batch_gt_x2: batch_img_x2, batch_gt_x4: batch_img_x4, batch_inputs: batch_in, is_training: True})
            info("at %d/%d, lr_: %.5f, g_loss: %.5f", epoch, step, lr_, loss_)

        if epoch % (epoches//5) == 0:
          model_name = "lapsrn-epoch-{}-step-{}-{}.ckpt".format(epoch, step, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
          saver.save(sess, os.path.join(g_ckpt_dir, model_name), global_step=step)
          info('save model at step: %d, in dir %s, name %s' %(step, g_ckpt_dir, model_name))

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  setup_project(FLAGS)

  print("===> Training")
  train(FLAGS)

if __name__ == '__main__':

  tf.app.run()
