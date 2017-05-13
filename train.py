from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import tensorflow as tf

from src.model import LapSRN
from src.utils import setup_project, sess_configure, tf_flag_setup, mse

# for log infos
pp = pprint.PrettyPrinter()
tf.logging.set_verbosity(tf.logging.INFO)
info = tf.logging.info

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS
tf_flag_setup(flags)

def train(graph, sess_conf, options):
  print('start to train')

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(FLAGS.gpu_id))):
      batch_gt_img, batch_input_img = read_images(path=FLAGS.dataset_dir, batch_size=FLAGS.batch_size)

      # define global variables
      counter = tf.get_variable(name="counter", shape=[], initializer=tf.constant_initializer(0), trainable=False)
      inputs = tf.placeholder(tf.float32, [1, 256, 256, 3])
      gt_imgs = tf.placeholder(tf.float32, [1, 256, 256, 3])

      model = LapSRN()
      model.forward(inputs)
      loss = model.l1_charbonnier_loss(gt_imgs)

      # loss
      lr = tf.train.exponential_decay(FLAGS.lr, counter, decay_rate=FLAGS.g_decay_rate, decay_steps=FLAGS.g_decay_steps, staircase=True)
      opt = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, momentum=0.9, epsilon=1e-8)
      # g_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
      grads = opt.compute_gradients(loss, var_list=model.vars)
      apply_gradient_opt = opt.apply_gradients(grads, global_step=counter)

      # restore generator
      all_variables = tf.global_variables()
      saver = tf.train.Saver(all_variables, max_to_keep=10)
      ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
      if ckpt and FLAGS.continued_training:
        saver.restore(sess, ckpt.model_checkpoint_path)
        info('restore the g from %s', ckpt.model_checkpoint_path)
        if FLAGS.debug:
          [print(v.name) for v in all_variables]
          print("all D variable" , len(all_variables))
          print("all global_variables" , len(tf.global_variables()))
          print("all local_variables" , len(tf.local_variables()))
      else:
        info('there is no ckpt for g...')
        sess.run(tf.variables_initializer(set(all_variables)))

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  setup_project(FLAGS)

  graph = tf.Graph()
  sess_conf = sess_configure()

  train(graph, sess_conf, FLAGS)

if __name__ == '__main__':
  tf.app.run()
