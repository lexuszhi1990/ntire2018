#!/usr/bin/python
'''
usage:

  python solver.py --dataset_dir=./dataset/train_291_coco_347_x2.h5 --continued_training=False --g_log_dir=./log/lapsrn-solver --g_ckpt_dir=./ckpt/lapsrn-solver --upscale_factor=4 --gpu_id=0 --lr=0.0006 --filter_num=128
  --g_decay_rate=0.9 --reg=0.0001 --epoches=20 --batch_size=8

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
  file_op = open(path,'w')

  for result in results:
    num = len(result[1])
    for l in range(num):

      scale = np.exp2(l+1).astype(int)
      file_op.write("for model %s, scale: %d, init lr: %f\n"%(result[0], scale, result[4]))
      file_op.write("average exec time: %.4fs;\tAaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n"%(np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])))
      print("scale: %d, init lr: %f\naverage exec time: %.4fs;\tAaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n"%(scale, result[4], np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])));

  file_op.close()

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS
tf_flag_setup(flags)

def main(_):

  pp = pprint.PrettyPrinter()
  pp.pprint(flags.FLAGS.__flags)

  print("===> Training")
  default_epoch = 6
  default_channel = 1
  default_sr_method = 'lapsrn-batch'
  test_dataset_path = './dataset/test/set5/mat'
  results = []

  lr_list = [FLAGS.lr, 0.0004, 0.0002, 0.0001]
  g_decay_rate_list = [FLAGS.g_decay_rate, 0.5]
  reg_list = [FLAGS.reg, 1e-5]
  # lr_list = [FLAGS.lr]
  # g_decay_rate_list = [FLAGS.g_decay_rate]
  # reg_list = [FLAGS.reg]

  for lr in lr_list:
    for decay_rate in g_decay_rate_list:
      for reg in reg_list:

        # training for one epoch
        model_list = []
        setup_project(FLAGS.dataset_dir, FLAGS.g_ckpt_dir, FLAGS.g_log_dir)
        for epoch in range(FLAGS.epoches):
          dataset = TrainDatasetFromHdf5(file_path=FLAGS.dataset_dir, batch_size=FLAGS.batch_size, upscale=FLAGS.upscale_factor)
          g_decay_steps = np.floor(np.log(decay_rate)/np.log(0.05) * (dataset.batch_ids*FLAGS.epoches*default_epoch))

          dataset.rebuild()
          del(dataset)

          model_path = model_list[-1] if len(model_list) != 0 else "None"
          model_list.append(train(FLAGS.batch_size, FLAGS.upscale_factor, default_epoch, lr, reg, FLAGS.filter_num, decay_rate, g_decay_steps, FLAGS.dataset_dir, FLAGS.g_ckpt_dir, FLAGS.g_log_dir, FLAGS.gpu_id, epoch!=0, model_path, FLAGS.debug))

        # model_list = ['./ckpt/lapsrn-solver/lapsrn-epoch-6-step-158-2017-06-21-02-37.ckpt-158']
        print("===> a training round ends, lr: %f, decay_rate: %f, reg: %f. The saved models are\n"%(lr, decay_rate, reg))
        print(model_list)

        # testing for one epoch
        for model_path in model_list:
          PSNR, SSIM, EXEC_TIME = SR(test_dataset_path, 2, FLAGS.upscale_factor, default_channel, FLAGS.filter_num, default_sr_method, model_path, FLAGS.gpu_id)
          results.append([model_path, PSNR, SSIM, EXEC_TIME, lr])

  print("===> Saving results")
  save_results(results)


if __name__ == '__main__':
  tf.app.run()
