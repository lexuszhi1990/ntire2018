#!/usr/bin/python
'''
usage:

  for v2:
  python solver.py --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v6 --g_ckpt_dir=./ckpt/lapser-solver_v6 --test_dataset_path=./dataset/test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1 --default_sr_method='lap_v6' --upscale_factor=4 --gpu_id=2 --filter_num=64 --batch_size=16

  for v3:
  # dataset 391x200, batch:12 => 15 min per epoch
  python solver.py --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v8 --g_ckpt_dir=./ckpt/lapser-solver_v8 --default_sr_method='lap_v8' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --gpu_id=3 --filter_num=64 --batch_size=12

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import argparse
import numpy as np
import tensorflow as tf

from train import train
from val import SR

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project


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

def setup_options():
  parser = argparse.ArgumentParser(description="LapSRN Test")
  parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
  parser.add_argument("--epoches", default=5, type=int, help="max epoches")
  parser.add_argument("--inner_epoches", default=6, type=int, help="inner epoches")
  parser.add_argument("--batch_size", default=2, type=int, help="batch size")
  parser.add_argument("--dataset_dir", default="null", type=str, help="image path")
  parser.add_argument("--g_ckpt_dir", default="null", type=str, help="g_ckpt_dir path")
  parser.add_argument("--g_log_dir", default="null", type=str, help="g_log_dir path")
  parser.add_argument("--default_sr_method", default="lapsrn", type=str, help="default_sr_method path")
  parser.add_argument("--test_dataset_path", default="null", type=str, help="test_dataset_path path")
  parser.add_argument('--debug', action='store_true', help='debug')
  parser.add_argument("--upscale_factor", default=4, type=int, help="scale factor, Default: 4")
  parser.add_argument("--filter_num", default=64, type=int, help="filter_num")
  parser.add_argument("--default_channel", default=1, type=int, help="default_channel")

  return parser

def main(_):

  parser = setup_options()
  opt = parser.parse_args()
  print(opt)

  inner_epoches = opt.inner_epoches
  default_channel = opt.default_channel
  default_sr_method = opt.default_sr_method
  test_dataset_path = opt.test_dataset_path
  gpu_id = opt.gpu_id
  epoches = opt.epoches
  batch_size = opt.batch_size
  dataset_dir = opt.dataset_dir
  g_ckpt_dir = opt.g_ckpt_dir
  g_log_dir = opt.g_log_dir
  debug = opt.debug
  upscale_factor = opt.upscale_factor
  filter_num = opt.filter_num

  results_file = "./tmp/results-{}-scale-{}-{}.txt".format(default_sr_method, upscale_factor, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
  f = open(results_file, 'w'); f.close()

  lr_list = [0.0004, 0.0003, 0.0002, 0.0001]
  g_decay_rate_list = [0.8, 0.7, 0.5, 0.1]
  reg_list = [5e-4, 1e-4]
  decay_final_rate_list = [0.05, 0.1]

  for lr in lr_list:
    for decay_rate in g_decay_rate_list:
      for reg in reg_list:
        for decay_final_rate in decay_final_rate_list:
          # training for one epoch
          model_list = []
          results = []

          print("===> Start Training for one parameters set")
          setup_project(dataset_dir, g_ckpt_dir, g_log_dir)
          for epoch in range(epoches):
            dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)
            g_decay_steps = np.floor(np.log(decay_rate)/np.log(decay_final_rate) * (dataset.batch_ids*epoches*inner_epoches))

            # dataset.rebuild()
            # del(dataset)

            model_path = model_list[-1] if len(model_list) != 0 else "None"
            saved_model = train(batch_size, upscale_factor, inner_epoches, lr, reg, filter_num, decay_rate, g_decay_steps, dataset_dir, g_ckpt_dir, g_log_dir, gpu_id, epoch!=0, model_path, debug)
            model_list.append(saved_model)

          print("===> Testing model")
          print(model_list)
          for model_path in model_list:
            PSNR, SSIM, EXEC_TIME = SR(test_dataset_path, 2, upscale_factor, default_channel, filter_num, default_sr_method, model_path, gpu_id)
            results.append([model_path, PSNR, SSIM, EXEC_TIME, lr, decay_rate, reg])

          print("===> a training round ends, lr: %f, decay_rate: %f, reg: %f. The saved models are\n"%(lr, decay_rate, reg))
          print("===> Saving results")
          save_results(results, results_file)


if __name__ == '__main__':
  tf.app.run()
