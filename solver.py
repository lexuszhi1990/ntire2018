#!/usr/bin/python
'''
usage:
  for v1:
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v1 --g_ckpt_dir=./ckpt/lapser-solver_v1 --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=2 --default_channel=1 --default_sr_method='LapSRN_v1' --upscale_factor=4 --filter_num=64 --batch_size=96

  for v2:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v2 --g_ckpt_dir=./ckpt/lapser-solver_v2 --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=2 --default_channel=1 --default_sr_method='LapSRN_v3' --upscale_factor=4 --filter_num=64 --batch_size=16

  for v3:
  # dataset 391x200, batch:12 => 15 min per epoch
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v3 --g_ckpt_dir=./ckpt/lapser-solver_v3 --default_sr_method='LapSRN_v3' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=2 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v4:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v4 --g_ckpt_dir=./ckpt/lapser-solver_v4 --default_sr_method='LapSRN_v4' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v5:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v5 --g_ckpt_dir=./ckpt/lapser-solver_v5 --default_sr_method='LapSRN_v5' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v6:
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v6 --g_ckpt_dir=./ckpt/lapser-solver_v6 --default_sr_method='LapSRN_v6' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=16

  for v7:
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/lapsrn-solver_v7 --g_ckpt_dir=./ckpt/lapser-solver_v7 --default_sr_method='LapSRN_v7' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/ntire_2k72.h5 --g_log_dir=./log/lapsrn-solver_v7 --g_ckpt_dir=./ckpt/lapser-solver_v7 --default_sr_method='LapSRN_v7' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  for v8:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v8 --g_ckpt_dir=./ckpt/lapser-solver_v8 --default_sr_method='LapSRN_v8' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v9:
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v9 --g_ckpt_dir=./ckpt/lapser-solver_v9 --default_sr_method='LapSRN_v9' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v10:
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v10 --g_ckpt_dir=./ckpt/lapser-solver_v10 --default_sr_method='LapSRN_v10' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for v11:
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v11 --g_ckpt_dir=./ckpt/lapser-solver_v11 --default_sr_method='LapSRN_v11' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=16

  for LapSRN_v13:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v13 --g_ckpt_dir=./ckpt/lapser-solver_v13 --default_sr_method='LapSRN_v13' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=8

  for LapSRN_v14:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/lapsrn-solver_v14_v1 --g_ckpt_dir=./ckpt/lapser-solver_v14 --default_sr_method='LapSRN_v14' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-solver_v14 --g_ckpt_dir=./ckpt/lapser-solver_v14 --default_sr_method='LapSRN_v14' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=2

  for v15, 16:
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v15 --g_ckpt_dir=./ckpt/lapser-solver_v15 --default_sr_method='LapSRN_v15' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v16 --g_ckpt_dir=./ckpt/lapser-solver_v16 --default_sr_method='LapSRN_v16' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v17 --g_ckpt_dir=./ckpt/lapser-solver_v17 --default_sr_method='LapSRN_v17' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2

  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/lapsrn-solver_v18 --g_ckpt_dir=./ckpt/lapser-solver_v18 --default_sr_method='LapSRN_v18' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2

  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v19 --g_ckpt_dir=./ckpt/lapser-solver_v19 --default_sr_method='LapSRN_v19' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2



  for v30:
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v30 --g_ckpt_dir=./ckpt/lapser-solver_v30 --default_sr_method='LapSRN_v30' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v31 --g_ckpt_dir=./ckpt/lapser-solver_v31 --default_sr_method='LapSRN_v31' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v32 --g_ckpt_dir=./ckpt/lapser-solver_v32 --default_sr_method='LapSRN_v32' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=2 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=16
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v33 --g_ckpt_dir=./ckpt/lapser-solver_v33 --default_sr_method='LapSRN_v33' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x2_x200.h5 --g_log_dir=./log/lapsrn-solver_v34 --g_ckpt_dir=./ckpt/lapser-solver_v34 --default_sr_method='LapSRN_v34' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4


For SR X8:
  for LapSRN_v41:
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v41 --g_ckpt_dir=./ckpt/lapser-LapSRN_v41 --default_sr_method='LapSRN_v41' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v42 --g_ckpt_dir=./ckpt/lapser-LapSRN_v42 --default_sr_method='LapSRN_v42' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v43 --g_ckpt_dir=./ckpt/lapser-LapSRN_v43 --default_sr_method='LapSRN_v43' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v44 --g_ckpt_dir=./ckpt/lapser-LapSRN_v44 --default_sr_method='LapSRN_v44' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v45 --g_ckpt_dir=./ckpt/lapser-LapSRN_v45 --default_sr_method='LapSRN_v45' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
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

from train import train
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
  results_pkl_file = "./tmp/results-{}-scale-{}-{}.pkl".format(default_sr_method, upscale_factor, time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
  f = open(results_file, 'w');
  import pdb
  pdb.set_trace()
  file_op.write(opt)
  f.close()

  pkl_results = []

  # for batch_size:32
  # hyper_params = [[0.00005, 0.1, 0.01, 1e-4], [0.0001, 0.1, 0.01, 1e-4], [0.00015, 0.50, 0.01, 1e-3], [0.0002, 0.70, 0.01, 1e-3], [0.00025, 0.90, 0.01, 1e-3]]
  # for batch_size:16
  # hyper_params = [[0.0001, 0.1, 0.05, 1e-4], [0.0001, 0.2, 0.01, 1e-4], [0.00015, 0.50, 0.01, 1e-3], [0.0002, 0.70, 0.01, 1e-3], [0.00025, 0.80, 0.01, 1e-3], [0.00035, 0.95, 0.01, 1e-3]]
  # for batch_size:8
  hyper_params = [[0.0001, 0.1, 0.05, 1e-4], [0.00015, 0.1, 0.01, 1e-4], [0.0002, 0.1, 0.05, 1e-4], [0.0002, 0.2, 0.01, 1e-4], [0.00025, 0.50, 0.01, 1e-3], [0.0003, 0.70, 0.01, 1e-3], [0.00035, 0.80, 0.01, 1e-3]]

  # step-num and residual-depth trade-off params
  # hyper_params = [[0.00015, 0.1, 0.01, 1e-4], [0.00025, 0.50, 0.01, 1e-3], [0.00035, 0.80, 0.01, 1e-3]]

  for lr, decay_rate, decay_final_rate, reg in hyper_params:
  # lr_list = [0.0003, 0.0004]
  # g_decay_rate_list = [0.2, 0.8]
  # reg_list = [1e-4]
  # decay_final_rate_list = [0.05, 0.01]
  # for reg in reg_list:
  #   for decay_final_rate in decay_final_rate_list:
  #     for decay_rate in g_decay_rate_list:
  #       for lr in lr_list:
          # training for one epoch
    model_list = []
    results = []

    print("===> Start Training for one parameters set")
    setup_project(dataset_dir, g_ckpt_dir, g_log_dir)
    for epoch in range(epoches):
      dataset = TrainDatasetFromHdf5(file_path=dataset_dir, batch_size=batch_size, upscale=upscale_factor)
      g_decay_steps = np.floor(np.log(decay_rate)/np.log(decay_final_rate) * (dataset.batch_ids*epoches*inner_epoches))

      model_path = model_list[-1] if len(model_list) != 0 else "None"
      saved_model = train(batch_size, upscale_factor, inner_epoches, lr, reg, filter_num, decay_rate, g_decay_steps, dataset_dir, g_ckpt_dir, g_log_dir, gpu_id, epoch!=0, default_sr_method, model_path, debug)
      model_list.append(saved_model)

    print("===> Testing model")
    print(model_list)
    for model_path in model_list:
      PSNR, SSIM, MSSSIM, EXEC_TIME = SR(test_dataset_path, 2, upscale_factor, default_channel, filter_num, default_sr_method, model_path, gpu_id)
      results.append([model_path, PSNR, SSIM, EXEC_TIME, lr, decay_rate, reg, decay_final_rate])
      pkl_results.append([model_path, PSNR, SSIM, EXEC_TIME, lr, decay_rate, reg, decay_final_rate])

    print("===> a training round ends, lr: %f, decay_rate: %f, reg: %f. The saved models are\n"%(lr, decay_rate, reg))
    print("===> Saving results")
    save_results(results, results_file, upscale_factor)

  print("===> Saving results to pkl at {}".format(results_pkl_file))
  pickle.dump(pkl_results, open(results_pkl_file, "w"))


if __name__ == '__main__':
  tf.app.run()
