from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import tensorflow as tf

from src.model import LapSRN
from src.dataset import DatasetFromHdf5
from src.utils import setup_project, sess_configure, tf_flag_setup, transform_reverse

# for log infos
pp = pprint.PrettyPrinter()
tf.logging.set_verbosity(tf.logging.INFO)
info = tf.logging.info

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS
tf_flag_setup(flags)

def train(graph, sess_conf, options):
  print('starts to train')

  # define training variables here
  batch_size = FLAGS.batch_size
  dataset_dir = FLAGS.dataset_dir
  g_ckpt_dir = FLAGS.g_ckpt_dir
  gpu_id = FLAGS.gpu_id
  g_decay_rate = FLAGS.g_decay_rate
  g_decay_steps = FLAGS.g_decay_steps
  upscale_factor = FLAGS.upscale_factor
  is_training_mode = FLAGS.is_training_mode
  continued_training = FLAGS.continued_training
  epoches = FLAGS.epoches
  g_log_dir = FLAGS.g_log_dir
  debug = FLAGS.debug

  lr = FLAGS.lr

  data_reader = DatasetFromHdf5(path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, 3])
      gt_imgs = tf.placeholder(tf.float32, [batch_size, None, None, 3])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN(inputs, gt_imgs, image_size=data_reader.input_image_size, is_training=is_training, upscale_factor=data_reader.upscale)
      model.extract_features()
      model.reconstruct()
      loss = model.l1_charbonnier_loss()

      upscaled_x2_img = transform_reverse(model.sr_imgs[0])
      upscaled_x4_img = transform_reverse(model.sr_imgs[1])
      g_output_sum = tf.summary.image("upscaled", upscaled_x4_img, max_outputs=2)
      gt_sum = tf.summary.image("gt_output", transform_reverse(gt_imgs), max_outputs=2)
      batch_input_sum = tf.summary.image("inputs", transform_reverse(inputs), max_outputs=2)
      gt_bicubic_sum = tf.summary.image("gt_bicubic_img", transform_reverse(tf.image.resize_images(inputs, size=[data_reader.gt_height, data_reader.gt_width], method=tf.image.ResizeMethod.BICUBIC, align_corners=False)), max_outputs=2)
      g_loss_sum = tf.summary.scalar("g_loss", loss)

      counter = tf.get_variable(name="counter", shape=[], initializer=tf.constant_initializer(0), trainable=False)
      lr = tf.train.exponential_decay(lr, counter, decay_rate=g_decay_rate, decay_steps=g_decay_steps, staircase=True)
      opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, momentum=0.9, epsilon=1e-8)
      # g_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
      grads = opt.compute_gradients(loss, var_list=model.vars)
      apply_gradient_opt = opt.apply_gradients(grads, global_step=counter)
      g_lr_sum = tf.summary.scalar("g_lr", lr)

      # restore model
      all_variables = tf.global_variables()
      saver = tf.train.Saver(all_variables, max_to_keep=30)
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

      for epoch in xrange(epoches):
        for step in xrange(1, data_reader.batch_ids):

          batch_inputs, batch_gt = data_reader.next(step-1)
          if step % 50 == 0:
            merged, apply_gradient_opt_, lr_, loss_ = sess.run([g_sum_all, apply_gradient_opt, lr, loss], feed_dict={gt_imgs: batch_gt, inputs: batch_inputs, is_training: is_training_mode})
            info("at %d/%d, lr_: %.5f, g_loss: %.5f", epoch, step, lr_, loss_)
            summary_writer.add_summary(merged, step + epoch*data_reader.batch_ids)
          else:
            apply_gradient_opt_, lr_, loss_ = sess.run([apply_gradient_opt, lr, loss], feed_dict={gt_imgs: batch_gt, inputs: batch_inputs, is_training: is_training_mode})
            info("at %d/%d, lr_: %.5f, g_loss: %.5f", epoch, step, lr_, loss_)

          if step % 50 == 0:
            model_name = "lapsrn-epoch-{}-step-{}-{}.ckpt".format(epoch, step, time.strftime('%y-%m-%d-%H-%M',time.localtime(time.time())))
            saver.save(sess, os.path.join(g_ckpt_dir, model_name), global_step=step)
            info('save model at step: %d, in dir %s, name %s' %(step, g_ckpt_dir, model_name))

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  setup_project(FLAGS)

  graph = tf.Graph()
  sess_conf = sess_configure()

  train(graph, sess_conf, FLAGS)

if __name__ == '__main__':
  # usage:
  #   python train.py --gpu_id=1 --epoches=2 --dataset_dir=./dataset/train_dataset_coco_selected.h5 --batch_size=16
  tf.app.run()
