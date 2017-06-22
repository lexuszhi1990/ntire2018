#!/usr/bin/python
'''
usage:

  python solver.py --dataset_dir=./dataset/train_x8.h5 --continued_training=False --g_log_dir=./log/lapsrn-solver_v2 --g_ckpt_dir=./ckpt/lapsrn-solver_v2 --g_decay_rate=0.5 --reg=0.0001 --epoches=10 --upscale_factor=4 --gpu_id=3 --filter_num=64 --lr=0.0002 --batch_size=32

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import numpy as np
import tensorflow as tf

from train import train
from val import SR

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project, tf_flag_setup


def save_results(results, path='./tmp/results.txt'):
  file_op = open(path,'a')

  for result in results:
    num = len(result[1])
    for l in range(num):

      scale = np.exp2(l+1).astype(int)
      file_op.write("for model %s, scale: %d, init lr: %f, decay_rate: %f, reg: %f\n"%(result[0], scale, result[4], result[5], result[6]))
      file_op.write("average exec time: %.4fs;\tAaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n\n"%(np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])))
      print("scale: %d, init lr: %f\naverage exec time: %.4fs;\tAaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n"%(scale, result[4], np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])));

  file_op.close()

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS
tf_flag_setup(flags)

def main(_):

  pp = pprint.PrettyPrinter()
  pp.pprint(flags.FLAGS.__flags)

  default_epoch = 1
  default_channel = 1
  default_sr_method = 'lapsrn-batch'
  test_dataset_path = './dataset/test/set5/mat'
  results_file = './tmp/results.txt'
  f = open(results_file, 'w'); f.close()

  lr_list = [0.0006, 0.0004, 0.0002, 0.0001]
  g_decay_rate_list = [0.9, 0.7, 0.5, 0.1]
  reg_list = [1e-4, 1e-5]
  # lr_list = [FLAGS.lr]
  # g_decay_rate_list = [FLAGS.g_decay_rate]
  # reg_list = [FLAGS.reg]


  for lr in lr_list:
    for decay_rate in g_decay_rate_list:
      for reg in reg_list:

        # training for one epoch
        model_list = []
        results = []

        print("===> Start Training for one parameters set")
        setup_project(FLAGS.dataset_dir, FLAGS.g_ckpt_dir, FLAGS.g_log_dir)
        for epoch in range(FLAGS.epoches):
          dataset = TrainDatasetFromHdf5(file_path=FLAGS.dataset_dir, batch_size=FLAGS.batch_size, upscale=FLAGS.upscale_factor)
          g_decay_steps = np.floor(np.log(decay_rate)/np.log(0.05) * (dataset.batch_ids*FLAGS.epoches*default_epoch))

          # dataset.rebuild()
          # del(dataset)

          model_path = model_list[-1] if len(model_list) != 0 else "None"
          saved_model = train(FLAGS.batch_size, FLAGS.upscale_factor, default_epoch, lr, reg, FLAGS.filter_num, decay_rate, g_decay_steps, FLAGS.dataset_dir, FLAGS.g_ckpt_dir, FLAGS.g_log_dir, FLAGS.gpu_id, epoch!=0, model_path, FLAGS.debug)
          model_list.append(saved_model)

        print("===> Testing model")
        print(model_list)
        # testing for one epoch
        for model_path in model_list:
          PSNR, SSIM, EXEC_TIME = SR(test_dataset_path, 2, FLAGS.upscale_factor, default_channel, FLAGS.filter_num, default_sr_method, model_path, FLAGS.gpu_id)
          results.append([model_path, PSNR, SSIM, EXEC_TIME, lr, decay_rate, reg])

        print("===> a training round ends, lr: %f, decay_rate: %f, reg: %f. The saved models are\n"%(lr, decay_rate, reg))
        print("===> Saving results")
        save_results(results, results_file)


if __name__ == '__main__':
  tf.app.run()
