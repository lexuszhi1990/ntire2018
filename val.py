#!/usr/bin/python
'''
usage:
  v2:
  python val.py --gpu_id=0 --channel=1 --filter_num=64 --sr_method=lapsrn_v2 --model=./saved_models/lapsrn/v2-31.10/lapsrn-epoch-5-step-724-2017-07-05-20-29.ckpt-724 --image=./dataset/mat_test/set5/mat --scale=4

  v3:
  python val.py --gpu_id=2 --channel=1 --filter_num=128 --sr_method=lapsrn_ml --model=./ckpt/lapser-solver_v8/lapsrn-epoch-2-step-1628-2017-07-13-13-43.ckpt-1628 --image=./dataset/mat_test/set5/mat --scale=4

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import numpy as np
from glob import glob

import scipy.io as sio

from src.cv2_utils import *


import tensorflow as tf

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4
from src.utils import sess_configure, trainsform, transform_reverse

from src.eval_dataset import eval_dataset
from src.evaluation import psnr as compute_psnr
from src.evaluation import _SSIMForMultiScale as compute_ssim

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

def generator(input_img, batch_size, scale, channel, filter_num, model_path, gpu_id):

  graph = tf.Graph()
  sess_conf = sess_configure(memory_per=.75)

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

      model = LapSRN_v3(inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size=img_size, upscale_factor=scale, filter_num=filter_num, is_training=is_training)
      model.init_gt_imgs()
      model.extract_features()
      model.reconstruct()
      upscaled_tf_img = model.upscaled_img(scale)

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
      print("\nIt takes {}s for processing".format(elapsed_time))

      return upscaled_img[0], elapsed_time

def cal_ssim(upscaled_img_y, gt_img_y):
  gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
  upscaled_img_ep = np.expand_dims(upscaled_img_y, axis=0)
  upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
  ssim = compute_ssim(gt_img_ep, upscaled_img_ep)[0]

  return ssim

def cal_image_index(gt_img_y, upscaled_img_y):

  upscaled_img_y = np.clip(upscaled_img_y*255., 0, 255.)
  gt_img_y = np.clip(gt_img_y*255., 0, 255.)

  psnr = compute_psnr(upscaled_img_y, gt_img_y)
  ssim = cal_ssim(upscaled_img_y, gt_img_y)

  return psnr, ssim

def SR(dataset_dir, batch_size, init_scale, channel, filter_num, sr_method, model_path, gpu_id):

  dataset_image_path = os.path.join(dataset_dir, '*.mat')

  PSNR = []
  SSIM = []
  EXEC_TIME = []
  scale_list = [2, 4, 8]

  for scale in scale_list[0:np.log2(init_scale).astype(int)]:
    ssims = []
    psnrs = []
    exec_time = []

    for filepath in glob(dataset_image_path):

      im_l_y, im_h_ycbcr, img_gt_y = load_img_from_mat(filepath, scale)
      im_h_y, elapsed_time = generator(im_l_y, batch_size, scale, channel, filter_num, model_path, gpu_id)
      save_mat(im_h_y, filepath, sr_method, scale)

      psnr, ssim = cal_image_index(img_gt_y, im_h_y[:,:,0])
      psnrs.append(psnr)
      ssims.append(ssim)
      exec_time.append(elapsed_time)

    PSNR.append(psnrs)
    SSIM.append(ssims)
    EXEC_TIME.append(exec_time)

  return PSNR, SSIM, EXEC_TIME


def setup_options():
  parser = argparse.ArgumentParser(description="LapSRN Test")
  parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
  parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
  parser.add_argument("--image", default="null", type=str, help="image path or single image")
  parser.add_argument("--output_dir", default="null", type=str, help="image path")
  parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
  parser.add_argument("--channel", default=1, type=int, help="input image channel, Default: 4")
  parser.add_argument("--sr_method", default="lapsrn", type=str, help="srn method")
  parser.add_argument("--batch_size", default=2, type=int, help="batch size")
  parser.add_argument("--filter_num", default=64, type=int, help="batch size")

  return parser

if __name__ == '__main__':

  scale_list = [2, 4, 8]
  parser = setup_options()
  opt = parser.parse_args()

  if not os.path.exists(opt.output_dir) and opt.output_dir != 'null':
    os.system('mkdir -p ' + opt.output_dir)

  if os.path.isdir(opt.image):

    PSNR, SSIM, EXEC_TIME = SR(opt.image, opt.batch_size, opt.scale, opt.channel, opt.filter_num, opt.sr_method, opt.model, opt.gpu_id)

    for scale in scale_list[0:np.log2(opt.scale).astype(int)]:
      l = np.log2(scale).astype(int) - 1
      print("for dataset %s, scale: %d, average exec time: %.4fs\n--Aaverage PSNR: %.4f;\tAaverage SSIM: %.4f\n"%(opt.image, scale, np.mean(EXEC_TIME[l]), np.mean(PSNR[l]), np.mean(SSIM[l])));

  else:
    print("please set correct input")

