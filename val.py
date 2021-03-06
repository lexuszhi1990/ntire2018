#!/usr/bin/python

import time
import argparse
import os
import numpy as np
from glob import glob
import scipy.io as sio
import shutil

import tensorflow as tf

from src.cv2_utils import *
from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19
from src.model import LapSRN_v2_v1, LapSRN_v2_v2
from src.model import LapSRN_v30, LapSRN_v31, LapSRN_v32, LapSRN_v33, LapSRN_v34
from src.model import LapSRN_v40, LapSRN_v41, LapSRN_v42, LapSRN_v43, LapSRN_v44
from src.model_new import EDSR_v100, EDSR_v101, EDSR_v102, EDSR_v103, EDSR_v104, EDSR_v105, EDSR_v106, EDSR_v107, EDSR_v108
from src.model_new import EDSR_v201, EDSR_v202, EDSR_v203, EDSR_v204, EDSR_v205, EDSR_v206, EDSR_v207, EDSR_v208, EDSR_v209, EDSR_v210, EDSR_v211, EDSR_v212, EDSR_v213, EDSR_v214, EDSR_v215, EDSR_v216, EDSR_v217, EDSR_v218, EDSR_v219, EDSR_v220, EDSR_v221, EDSR_v222, EDSR_v223, EDSR_v224, EDSR_v225, EDSR_v226, EDSR_v227, EDSR_v228, EDSR_v229, EDSR_v230, EDSR_v241, EDSR_v242, EDSR_v243, EDSR_v244, EDSR_v245, EDSR_v246, EDSR_v247, EDSR_v248, EDSR_v249, EDSR_v250, EDSR_v251, EDSR_v252, EDSR_v253, EDSR_v254, EDSR_v255, EDSR_v256
# from sr.model_new import EDSR_v231, EDSR_v232, EDSR_v233, EDSR_v234, EDSR_v235, EDSR_v236, EDSR_v237, EDSR_v238, EDSR_v239, EDSR_v240
from src.model_new import EDSR_v301, EDSR_v302, EDSR_v303, EDSR_v304, EDSR_v305, EDSR_v306, EDSR_v307, EDSR_v308, EDSR_v309, EDSR_v310, EDSR_v311, EDSR_v312, EDSR_v313, EDSR_v314, EDSR_v315, EDSR_v316, EDSR_v321, EDSR_v322, EDSR_v323, EDSR_v324, EDSR_v325, EDSR_v326, EDSR_v327, EDSR_v328, EDSR_v329, EDSR_v330

from src.gan import SRGAN, EDSR_v401
from src.model_new import EDSR_LFW_v1, EDSR_LFW_v2, EDSR_LFW_v3, EDSR_LFW_v4, EDSR_LFW_v5, EDSR_LFW_v6

from src.model_new import SRGAN_x2, SRGAN_x2_v1, SRGAN_x2_v2, SRGAN_x4, SRGAN_x4_v1, SRGAN_x4_v2, SRGAN_x8, SRGAN_x8_v1, SRGAN_x8_v2
from src.model_new import LapSRN_baseline_x2, LapSRN_baseline_x2_v1, LapSRN_baseline_x2_v2, LapSRN_baseline_x4, LapSRN_baseline_x4_v1, LapSRN_baseline_x4_v2, LapSRN_baseline_x8, LapSRN_baseline_x8_v1, LapSRN_baseline_x8_v2

from src.model_new import EDSR_V500 , EDSR_V501 , EDSR_V510, EDSR_V511 , EDSR_V512

from src.utils import sess_configure, trainsform, transform_reverse

from src.eval_dataset import eval_dataset
from src.evaluation import shave_bd, compute_psnr, compute_ssim, compute_msssim

def load_img_from_mat(img_mat_path, scale):
  image_hash = sio.loadmat(img_mat_path)

  im_l_y = image_hash['label_x{}_y'.format(8//scale)]
  im_bicubic_ycbcr = image_hash['bicubic_l{}_x{}_ycbcr'.format(scale, scale)]
  im_bicubic_ycbcr = np.clip(im_bicubic_ycbcr*255., 0, 255.)
  img_gt = image_hash['label_x8_y']

  return im_l_y, im_bicubic_ycbcr, img_gt

def val_img_path(img_path, scale, sr_method, output_dir):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_l{}_{}_x{}.png".format(img_name, scale, sr_method, str(scale))
  if output_dir != 'null' and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name)

def save_img(image, img_path, scale, sr_method, output_dir):
  output_img_path = val_img_path(img_path)
  imsave(output_img_path, image)

  print("upscaled image size {}".format(np.shape(image)))
  print("save image at {}".format(output_img_path))

def save_mat(img, path, sr_method, scale):
  image_hash = sio.loadmat(path)
  img_key = '{}_l{}_x{}_y'.format(sr_method, scale, scale)
  image_hash[img_key] = img
  sio.savemat(path, image_hash)

  print('save mat at {} in {}'.format(path, img_key))

def matlab_validation(dataset_dir, sr_method, scale):
  dataset_dir_list = dataset_dir.split('/')[0:-1]
  base_dataset_dir = '/'.join(dataset_dir_list)

  eng = matlab.engine.start_matlab()

  eng.addpath(r'./src/evaluation_mat', nargout=0)
  eng.addpath(r'./src/evaluation_mat/ifc-drrn')
  eng.addpath(r'./src/evaluation_mat/matlabPyrTools')

  eng.eval_dataset_mat(base_dataset_dir, 'lapsrn/mat', sr_method, scale)
  eng.eval_dataset(base_dataset_dir, sr_method, scale)

  eng.quit()

def load_models(sr_method, model_path):
  # os.system('scp youlei@219.223.251.241:/home/youlei/workplace/srn_bishe/ckpt/EDSR_v215/EDSR_v215-epoch-1-step-19548-2017-10-12-13-44.ckpt-19548.index ./')

  # g_dir = './ckpt/' + sr_method
  g_dir = '/'.join(model_path.split('/')[:-1])
  print("load model from {}".format(g_dir))
  if tf.gfile.Exists(g_dir):
    # tf.gfile.DeleteRecursively(g_dir)
    return
  else:
    tf.gfile.MakeDirs(g_dir)

    # command = os.path.join('scp youlei@219.223.251.241:/home/youlei/workplace/srn_face/', model_path)
    command = os.path.join('scp youlei@219.223.251.241:/home/youlei/workplace/srn_bishe', model_path)
    os.system(command + '.index ' + g_dir)
    os.system(command + '.meta ' + g_dir)
    os.system(command + '.data-00000-of-00001 ' + g_dir)
    print(command)

def generator(input_img, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id):

  graph = tf.Graph()
  sess_conf = sess_configure(memory_per=.95)

  img_size = input_img.shape
  height, width = input_img.shape
  batch_images = np.zeros((batch_size, height, width, channel))
  batch_images[0, :, :, 0] = input_img

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x2 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x4 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x8 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      is_training = tf.placeholder(tf.bool, [])

      SRNet = globals()[model_name]
      model = SRNet(inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size=img_size, upscale_factor=scale, filter_num=filter_num, is_training=is_training)
      model.init_gt_imgs()
      model.extract_features()
      model.reconstruct()
      upscaled_tf_img = model.get_image(scale)

      saver = tf.train.Saver()
      if os.path.isdir(model_path):
        latest_ckpt = tf.train.latest_checkpoint(model_path)
        if latest_ckpt:
          saver.restore(sess, latest_ckpt)
          print("restore model from dir %s"%latest_ckpt)
        else:
          print("cannot restore model from %s, plz checkout"%latest_ckpt)
      else:
        saver.restore(sess, model_path)
        print("restore model from file %s"%model_path)

      start_time = time.time()
      upscaled_img = sess.run(upscaled_tf_img, feed_dict={inputs: batch_images, is_training: False})
      elapsed_time = time.time() - start_time

      return upscaled_img[0], elapsed_time

def cal_ssim(upscaled_img_y, gt_img_y):
  gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
  upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
  ssim = compute_ssim(gt_img_ep, upscaled_img_ep)

  return ssim

def cal_msssim(upscaled_img_y, gt_img_y):
  gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
  upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
  msssim = compute_msssim(gt_img_ep, upscaled_img_ep)

  return msssim

def cal_image_index(gt_img_y, upscaled_img_y, scale):

  upscaled_img_y = np.clip(upscaled_img_y*255., 0, 255.)
  gt_img_y = np.clip(gt_img_y*255., 0, 255.)

  upscaled_img_y = shave_bd(upscaled_img_y, scale)
  gt_img_y = shave_bd(gt_img_y, scale)

  psnr = compute_psnr(upscaled_img_y, gt_img_y)
  ssim = cal_ssim(upscaled_img_y, gt_img_y)
  msssim = cal_msssim(upscaled_img_y, gt_img_y)

  return psnr, ssim, msssim

def SR(dataset_dir, batch_size, scale, channel, filter_num, sr_method, model_path, gpu_id):

  dataset_image_path = os.path.join(dataset_dir, '*.mat')

  PSNR = []
  SSIM = []
  MSSSIM = []
  EXEC_TIME = []

  # scale_list = [2, 4, 8]
  # for scale in scale_list[0:np.log2(init_scale).astype(int)]:
  psnrs = []
  ssims = []
  msssims = []
  exec_time = []

  for filepath in glob(dataset_image_path):

    tf.reset_default_graph()

    im_l_y, im_h_ycbcr, img_gt_y = load_img_from_mat(filepath, scale)
    im_h_y, elapsed_time = generator(im_l_y, batch_size, scale, channel, filter_num, sr_method, model_path, gpu_id)
    save_mat(im_h_y, filepath, sr_method, scale)

    psnr, ssim, msssim = cal_image_index(img_gt_y, im_h_y[:,:,0], scale)
    psnrs.append(psnr)
    ssims.append(ssim)
    msssims.append(msssim)
    exec_time.append(elapsed_time)

    print("for image %s, scale: %d, average exec time: %.4fs\n-- PSNR/SSIM/MSSSIM: %.4f/%.4f/%.4f\n"%(filepath, scale, elapsed_time, psnr, ssim, msssim))

  PSNR.append(psnrs)
  SSIM.append(ssims)
  MSSSIM.append(msssims)
  EXEC_TIME.append(exec_time)

  return PSNR, SSIM, MSSSIM, EXEC_TIME


def setup_options():
  parser = argparse.ArgumentParser(description="LapSRN Test")
  parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
  parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
  parser.add_argument("--image", default="null", type=str, help="image path or single image")
  parser.add_argument("--output_dir", default="null", type=str, help="image path")
  parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
  parser.add_argument("--channel", default=1, type=int, help="input image channel, Default: 4")
  parser.add_argument("--sr_method", default="lapsrn", type=str, help="srn method")
  parser.add_argument("--batch_size", default=1, type=int, help="batch size")
  parser.add_argument("--filter_num", default=64, type=int, help="batch size")
  parser.add_argument("--matlab_val", action="store_true", help="use matlab code validation in the end...")
  parser.add_argument("--validate_all", action="store_true", help="use matlab code validation in the end...")

  return parser

if __name__ == '__main__':
  scale_list = [2, 4, 8]
  parser = setup_options()
  opt = parser.parse_args()

  load_models(opt.sr_method, opt.model)

  if opt.validate_all:

    # dataset_dir_list = ['./dataset/mat_test/set5/mat', './dataset/mat_test/set14/mat', './dataset/mat_test/bsd100/mat', './dataset/mat_test/urban100/mat', './dataset/mat_test/manga109/mat']
    dataset_dir_list = ['./dataset/mat_test/set5/mat', './dataset/mat_test/set14/mat', './dataset/mat_test/bsd100/mat']
    for test_dir in dataset_dir_list:
      PSNR, SSIM, MSSSIM, EXEC_TIME = SR(test_dir, opt.batch_size, opt.scale, opt.channel, opt.filter_num, opt.sr_method, opt.model, opt.gpu_id)

      # for scale in scale_list[0:np.log2(opt.scale).astype(int)]:
      # l = np.log2(opt.scale).astype(int) - 1
      print("for dataset %s, scale: %d, average exec time: %.4fs\n--Aaverage PSNR: %.4f;\tAaverage SSIM: %.4f;\tAaverage MSSSIM: %.4f\n"%(test_dir, opt.scale, np.mean(EXEC_TIME[0]), np.mean(PSNR[0]), np.mean(SSIM[0]), np.mean(MSSSIM[0])));

      if opt.matlab_val:
        import matlab
        import matlab.engine

        matlab_validation(test_dir, opt.sr_method, opt.scale)

        # copy results to saved_models
        dataset_dir_list = test_dir.split('/')[0:-1]
        test_dataset=dataset_dir_list[-1]
        base_dataset_dir = '/'.join(dataset_dir_list)
        dst_path = 'saved_models/x{}/{}'.format(opt.scale, opt.sr_method)
        if not os.path.exists(dst_path):
          os.mkdir(dst_path)

        shutil.copytree('{}/lapsrn/{}'.format(base_dataset_dir, opt.sr_method), '{}/{}'.format(dst_path, test_dataset))
        os.system('cp {} {}'.format(opt.model+'*', dst_path))
        print('copied results to saved_models: {}'.format('{}/{}'.format(base_dataset_dir, opt.sr_method)))
  else:

    if os.path.isdir(opt.image):
      PSNR, SSIM, MSSSIM, EXEC_TIME = SR(opt.image, opt.batch_size, opt.scale, opt.channel, opt.filter_num, opt.sr_method, opt.model, opt.gpu_id)

      # for scale in scale_list[0:np.log2(opt.scale).astype(int)]:
      # l = np.log2(opt.scale).astype(int) - 1
      print("for dataset %s, scale: %d, average exec time: %.4fs\n--Aaverage PSNR: %.4f;\tAaverage SSIM: %.4f;\tAaverage MSSSIM: %.4f\n"%(opt.image, opt.scale, np.mean(EXEC_TIME[0]), np.mean(PSNR[0]), np.mean(SSIM[0]), np.mean(MSSSIM[0])));

      if opt.matlab_val:
        import matlab
        import matlab.engine

        matlab_validation(opt.image, opt.sr_method, opt.scale)

    else:
      print("please set correct input")

