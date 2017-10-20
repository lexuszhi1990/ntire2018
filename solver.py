# -*- coding: utf-8 -*-
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

  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v44 --g_ckpt_dir=./ckpt/lapser-LapSRN_v44 --default_sr_method='LapSRN_v44' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=2
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/lapsrn-LapSRN_v45 --g_ckpt_dir=./ckpt/lapser-LapSRN_v45 --default_sr_method='LapSRN_v45' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=2

  for contrast:

  model EDSR_v100
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-solver_v100 --g_ckpt_dir=./ckpt/edsr-solver_v100 --default_sr_method='EDSR_v100' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-solver_v101 --g_ckpt_dir=./ckpt/edsr-solver_v101 --default_sr_method='EDSR_v101' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-solver_v102 --g_ckpt_dir=./ckpt/edsr-solver_v102 --default_sr_method='EDSR_v102' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-solver_v103 --g_ckpt_dir=./ckpt/edsr-solver_v103 --default_sr_method='EDSR_v103' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v104 --g_ckpt_dir=./ckpt/EDSR_v104 --default_sr_method='EDSR_v104' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v105 --g_ckpt_dir=./ckpt/EDSR_v105 --default_sr_method='EDSR_v105' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v106 --g_ckpt_dir=./ckpt/EDSR_v106 --default_sr_method='EDSR_v106' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  # 放大倍数为2, 相同的子级联深度，不同的级联总数
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-solver_v201 --g_ckpt_dir=./ckpt/edsr-solver_v201 --default_sr_method='EDSR_v201' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v202 --g_ckpt_dir=./ckpt/edsr-EDSR_v202 --default_sr_method='EDSR_v202' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v203 --g_ckpt_dir=./ckpt/edsr-EDSR_v203 --default_sr_method='EDSR_v203' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v204 --g_ckpt_dir=./ckpt/edsr-EDSR_v204 --default_sr_method='EDSR_v204' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v205 --g_ckpt_dir=./ckpt/edsr-EDSR_v205 --default_sr_method='EDSR_v205' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v206 --g_ckpt_dir=./ckpt/edsr-EDSR_v206 --default_sr_method='EDSR_v206' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v207 --g_ckpt_dir=./ckpt/edsr-EDSR_v207 --default_sr_method='EDSR_v207' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v208 --g_ckpt_dir=./ckpt/edsr-EDSR_v208 --default_sr_method='EDSR_v208' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v209 --g_ckpt_dir=./ckpt/edsr-EDSR_v209 --default_sr_method='EDSR_v209' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/edsr-EDSR_v210 --g_ckpt_dir=./ckpt/edsr-EDSR_v210 --default_sr_method='EDSR_v210' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4

  # 放大倍数为4, 相同的子级联深度，不同的级联总数
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v211 --g_ckpt_dir=./ckpt/EDSR_v211 --default_sr_method='EDSR_v211' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v212 --g_ckpt_dir=./ckpt/EDSR_v212 --default_sr_method='EDSR_v212' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v213 --g_ckpt_dir=./ckpt/EDSR_v213 --default_sr_method='EDSR_v213' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v214 --g_ckpt_dir=./ckpt/EDSR_v214 --default_sr_method='EDSR_v214' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v215 --g_ckpt_dir=./ckpt/EDSR_v215 --default_sr_method='EDSR_v215' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v216 --g_ckpt_dir=./ckpt/EDSR_v216 --default_sr_method='EDSR_v216' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v217 --g_ckpt_dir=./ckpt/EDSR_v217 --default_sr_method='EDSR_v217' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v218 --g_ckpt_dir=./ckpt/EDSR_v218 --default_sr_method='EDSR_v218' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v219 --g_ckpt_dir=./ckpt/EDSR_v219 --default_sr_method='EDSR_v219' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v220 --g_ckpt_dir=./ckpt/EDSR_v220 --default_sr_method='EDSR_v220' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=32 --batch_size=4

  # 放大倍数为8, 相同的子级联深度，不同的级联总数
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v221 --g_ckpt_dir=./ckpt/EDSR_v221 --default_sr_method='EDSR_v221' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v222 --g_ckpt_dir=./ckpt/EDSR_v222 --default_sr_method='EDSR_v222' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v223 --g_ckpt_dir=./ckpt/EDSR_v223 --default_sr_method='EDSR_v223' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v224 --g_ckpt_dir=./ckpt/EDSR_v224 --default_sr_method='EDSR_v224' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v225 --g_ckpt_dir=./ckpt/EDSR_v225 --default_sr_method='EDSR_v225' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v226 --g_ckpt_dir=./ckpt/EDSR_v226 --default_sr_method='EDSR_v226' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v227 --g_ckpt_dir=./ckpt/EDSR_v227 --default_sr_method='EDSR_v227' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v228 --g_ckpt_dir=./ckpt/EDSR_v228 --default_sr_method='EDSR_v228' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v229 --g_ckpt_dir=./ckpt/EDSR_v229 --default_sr_method='EDSR_v229' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v230 --g_ckpt_dir=./ckpt/EDSR_v230 --default_sr_method='EDSR_v230' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4

  # 放大倍数为2, 相同的级联总数，不同的子级联深度
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v241 --g_ckpt_dir=./ckpt/EDSR_v241 --default_sr_method='EDSR_v241' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v242 --g_ckpt_dir=./ckpt/EDSR_v242 --default_sr_method='EDSR_v242' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v243 --g_ckpt_dir=./ckpt/EDSR_v243 --default_sr_method='EDSR_v243' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v244 --g_ckpt_dir=./ckpt/EDSR_v244 --default_sr_method='EDSR_v244' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v245 --g_ckpt_dir=./ckpt/EDSR_v245 --default_sr_method='EDSR_v245' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4

  # 放大倍数为4, 相同的级联总数，不同的子级联深度
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v246 --g_ckpt_dir=./ckpt/EDSR_v246 --default_sr_method='EDSR_v246' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v247 --g_ckpt_dir=./ckpt/EDSR_v247 --default_sr_method='EDSR_v247' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v248 --g_ckpt_dir=./ckpt/EDSR_v248 --default_sr_method='EDSR_v248' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v249 --g_ckpt_dir=./ckpt/EDSR_v249 --default_sr_method='EDSR_v249' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v250 --g_ckpt_dir=./ckpt/EDSR_v250 --default_sr_method='EDSR_v250' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1 --upscale_factor=4 --filter_num=64 --batch_size=4

  # 放大倍数为8, 相同的级联总数，不同的子级联深度
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v251 --g_ckpt_dir=./ckpt/EDSR_v251 --default_sr_method='EDSR_v251' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v252 --g_ckpt_dir=./ckpt/EDSR_v252 --default_sr_method='EDSR_v252' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v253 --g_ckpt_dir=./ckpt/EDSR_v253 --default_sr_method='EDSR_v253' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v254 --g_ckpt_dir=./ckpt/EDSR_v254 --default_sr_method='EDSR_v254' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v255 --g_ckpt_dir=./ckpt/EDSR_v255 --default_sr_method='EDSR_v255' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=16 --batch_size=4

  # 放大倍数为8, 级联总数为8，每级深度为10，不同长度的Expand-squeeze 大小
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v321 --g_ckpt_dir=./ckpt/EDSR_v321 --default_sr_method='EDSR_v321' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v322 --g_ckpt_dir=./ckpt/EDSR_v322 --default_sr_method='EDSR_v322' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v323 --g_ckpt_dir=./ckpt/EDSR_v323 --default_sr_method='EDSR_v323' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v324 --g_ckpt_dir=./ckpt/EDSR_v324 --default_sr_method='EDSR_v324' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v325 --g_ckpt_dir=./ckpt/EDSR_v325 --default_sr_method='EDSR_v325' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v326 --g_ckpt_dir=./ckpt/EDSR_v326 --default_sr_method='EDSR_v326' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v328 --g_ckpt_dir=./ckpt/EDSR_v328 --default_sr_method='EDSR_v328' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v329 --g_ckpt_dir=./ckpt/EDSR_v329 --default_sr_method='EDSR_v329' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/mat_train_391_x200.h5 --g_log_dir=./log/EDSR_v330 --g_ckpt_dir=./ckpt/EDSR_v330 --default_sr_method='EDSR_v330' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=8 --fnilter_num=64 --batch_size=4

  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/LFW_SR_train_36.h5 --g_log_dir=./log/EDSR_LFW_v1 --g_ckpt_dir=./ckpt/EDSR_LFW_v1 --default_sr_method='EDSR_LFW_v1' --test_dataset_path=./dataset/test/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=1 python solver.py --gpu_id=1 --dataset_dir=./dataset/LFW_SR_train_36.h5 --g_log_dir=./log/EDSR_LFW_v2 --g_ckpt_dir=./ckpt/EDSR_LFW_v2 --default_sr_method='EDSR_LFW_v2' --test_dataset_path=./dataset/test/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=2 --filter_num=64 --batch_size=4
  CUDA_VISIBLE_DEVICES=2 python solver.py --gpu_id=2 --dataset_dir=./dataset/LFW_SR_train_36.h5 --g_log_dir=./log/EDSR_LFW_v3 --g_ckpt_dir=./ckpt/EDSR_LFW_v3 --default_sr_method='EDSR_LFW_v3' --test_dataset_path=./dataset/test/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=128 --batch_size=4
  CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/LFW_SR_train_36.h5 --g_log_dir=./log/EDSR_LFW_v4 --g_ckpt_dir=./ckpt/EDSR_LFW_v4 --default_sr_method='EDSR_LFW_v4' --test_dataset_path=./dataset/test/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=128 --batch_size=4

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
  f = open(results_file, 'w')
  f.write(str(opt))
  f.close()

  pkl_results = []

  # for batch_size:32
  # hyper_params = [[0.00005, 0.1, 0.01, 1e-4], [0.0001, 0.1, 0.01, 1e-4], [0.00015, 0.50, 0.01, 1e-3], [0.0002, 0.70, 0.01, 1e-3], [0.00025, 0.90, 0.01, 1e-3]]
  # for batch_size:16
  # hyper_params = [[0.0001, 0.1, 0.05, 1e-4], [0.0001, 0.2, 0.01, 1e-4], [0.00015, 0.50, 0.01, 1e-3], [0.0002, 0.70, 0.01, 1e-3], [0.00025, 0.80, 0.01, 1e-3], [0.00035, 0.95, 0.01, 1e-3]]
  # for batch_size:8
  # lr_list = [0.0003, 0.0004]
  # g_decay_rate_list = [0.2, 0.8]
  # reg_list = [1e-4]
  # decay_final_rate_list = [0.05, 0.01]
  # for reg in reg_list:
  #   for decay_final_rate in decay_final_rate_list:
  #     for decay_rate in g_decay_rate_list:
  #       for lr in lr_list:

  # for k207 with these params:
    # CUDA_VISIBLE_DEVICES=0 python solver.py --gpu_id=0 --dataset_dir=./dataset/mat_train_391_x4_x200.h5 --g_log_dir=./log/EDSR_v106 --g_ckpt_dir=./ckpt/EDSR_v106 --default_sr_method='EDSR_v106' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1  --upscale_factor=4 --filter_num=64 --batch_size=4
  # archived best results with [0.0002, 0.1, 0.05, 1e-4]
  # hyper_params = [[0.0001, 0.1, 0.05, 1e-4], [0.00015, 0.1, 0.01, 1e-4], [0.0002, 0.1, 0.05, 1e-4], [0.0002, 0.2, 0.01, 1e-4], [0.00025, 0.50, 0.01, 1e-3], [0.0003, 0.70, 0.01, 1e-3], [0.00035, 0.80, 0.01, 1e-3]]

  hyper_params = [[0.0002, 0.1, 0.05, 1e-4], [0.0002, 0.1, 0.05, 1e-3], [0.0002, 0.1, 0.01, 1e-4], [0.0004, 0.5, 0.01, 1e-4]]

  # step-num and residual-depth trade-off params
  # hyper_params = [[0.00015, 0.1, 0.01, 1e-4], [0.00025, 0.50, 0.01, 1e-3], [0.00035, 0.80, 0.01, 1e-3]]

  for lr, decay_rate, decay_final_rate, reg in hyper_params:
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
