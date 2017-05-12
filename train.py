from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import tensorflow as tf

from src.model import LapSRN
from src.utils import setup_project, sess_configure, tf_flag_setup

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

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  setup_project(FLAGS)

  graph = tf.Graph()
  sess_conf = sess_configure()

  train(graph, sess_conf, FLAGS)

if __name__ == '__main__':
  tf.app.run()
