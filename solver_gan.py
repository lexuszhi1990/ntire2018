# -*- coding: utf-8 -*-
#!/usr/bin/python

'''

CUDA_VISIBLE_DEVICES=3 python solver_gan.py --gpu_id=3 --dataset_dir=./dataset/LFW_SR_v1_36.h5 --g_log_dir=./log/EDSR_LFW_v4_wgan --g_ckpt_dir=./ckpt/EDSR_LFW_v4_wgan --default_sr_method='EDSR_LFW_v4' --test_dataset_path=./dataset/test_1/mat --epoches=1 --inner_epoches=1 --default_channel=1 --is_wgan --upscale_factor=4 --g_filter_num=64 --d_filter_num=64 --batch_size=4

CUDA_VISIBLE_DEVICES=2 python solver_gan.py --gpu_id=2 --dataset_dir=./dataset/LFW_SR_x8_v1_36.h5 --g_log_dir=./log/EDSR_LFW_v5_wgan --g_ckpt_dir=./ckpt/EDSR_LFW_v5_wgan --default_sr_method='EDSR_LFW_v5' --test_dataset_path=./dataset/test_1/mat --epoches=1 --inner_epoches=1 --default_channel=1 --is_wgan --upscale_factor=8 --g_filter_num=64 --d_filter_num=64 --batch_size=4

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import argparse
import numpy as np
import cPickle as pickle
import tensorflow as tf

from gan_train import train
from val import SR

from src.dataset import TrainDatasetFromHdf5
from src.utils import setup_project

def save_results(results, path='./tmp/results.txt', scale=4):
  file_op = open(path,'a')

  for result in results:
    num = len(result[1])
    for l in range(num):

      file_op.write("for model %s, scale: %d, init lr: %f, decay_rate: %f, reg: %f, decay_final_rate: %f\n"%(result[0], scale, result[4], result[5], result[6], result[7]))
      file_op.write("average exec time: %.4fs;\tAaverage PSNR/SSIM: %.4f/%.4f\n\n"%(np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])))
      print("scale: %d, init lr: %f\naverage exec time: %.4fs;\tAaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n"%(scale, result[4], np.mean(result[3][l]), np.mean(result[1][l]), np.mean(result[2][l])));

  file_op.close()

def setup_options():
  parser = argparse.ArgumentParser(description="LapSRN Test")
  parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
  parser.add_argument("--epoches", default=5, type=int, help="max epoches")
  parser.add_argument("--inner_epoches", default=1, type=int, help="inner epoches")
  parser.add_argument("--batch_size", default=2, type=int, help="batch size")
  parser.add_argument("--dataset_dir", default="null", type=str, help="image path")
  parser.add_argument("--g_ckpt_dir", default="null", type=str, help="g_ckpt_dir path")
  parser.add_argument("--g_log_dir", default="null", type=str, help="g_log_dir path")
  parser.add_argument("--default_sr_method", default="lapsrn", type=str, help="default_sr_method path")
  parser.add_argument("--test_dataset_path", default="null", type=str, help="test_dataset_path path")
  parser.add_argument('--is_wgan', action='store_true', help='debug')
  parser.add_argument("--upscale_factor", default=4, type=int, help="scale factor, Default: 4")
  parser.add_argument("--g_filter_num", default=64, type=int, help="filter_num")
  parser.add_argument("--d_filter_num", default=64, type=int, help="filter_num")
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
  upscale_factor = opt.upscale_factor
  g_filter_num = opt.g_filter_num
  d_filter_num = opt.d_filter_num
  default_channel = opt.default_channel
  is_wgan = opt.is_wgan

  results_file = "./tmp/results-{}-scale-{}-{}.txt".format(default_sr_method, upscale_factor, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
  results_pkl_file = "./tmp/results-{}-scale-{}-{}.pkl".format(default_sr_method, upscale_factor, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
  f = open(results_file, 'w')
  f.write(str(opt))
  f.close()

  pkl_results = []

  hyper_params = [[0.0002, 0.0002, 0.1, 0.1, 0.05, 1e-4], [0.0001, 0.0001, 0.1, 0.1, 0.05, 1e-4], [0.0002, 0.0002, 0.8, 0.8, 0.05, 1e-4]]

  for g_lr, d_lr, g_decay_rate, d_decay_rate, decay_final_rate, reg in hyper_params:
    model_list = []
    results = []

    print("===> Start Training for one parameters set")
    setup_project(dataset_dir, g_ckpt_dir, g_log_dir)

    dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)
    g_decay_steps = np.floor(np.log(g_decay_rate)/np.log(decay_final_rate) * (dataset.batch_ids*inner_epoches))
    d_decay_steps = np.floor(np.log(d_decay_rate)/np.log(decay_final_rate) * (dataset.batch_ids*inner_epoches))

    saved_model = train(g_log_dir, gpu_id, g_ckpt_dir, dataset_dir, default_sr_method, batch_size, upscale_factor, default_channel, g_decay_steps, d_decay_steps, g_filter_num, d_filter_num, g_lr, d_lr, g_decay_rate, d_decay_rate, is_wgan, True)
    model_list.append(saved_model)

    print("===> Testing model")
    print(model_list)

    for model_path in model_list:
      PSNR, SSIM, MSSSIM, EXEC_TIME = SR(test_dataset_path, 2, upscale_factor, default_channel, g_filter_num, default_sr_method, model_path, gpu_id)
      results.append([model_path, PSNR, SSIM, EXEC_TIME, g_lr, d_lr, g_decay_rate, d_decay_rate, reg, decay_final_rate])
      pkl_results.append([model_path, PSNR, SSIM, EXEC_TIME, g_lr, d_lr, g_decay_rate, d_decay_rate, reg, decay_final_rate])

    print("===> a training round ends, g_lr: %f, d_lr: %f, g_decay_rate: %f, d_decay_rate: %f, reg: %f. The saved models are\n"%(g_lr, d_lr, g_decay_rate, d_decay_rate, reg))
    print("===> Saving results")
    save_results(results, results_file, upscale_factor)

  print("===> Saving results to pkl at {}".format(results_pkl_file))
  pickle.dump(pkl_results, open(results_pkl_file, "w"))


if __name__ == '__main__':
  tf.app.run()
