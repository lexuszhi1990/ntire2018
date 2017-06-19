#!/usr/bin/python
'''
usage:

  python val.py --gpu_id=4 --channel=1 --model=./ckpt/lapsrn/lapsrn-epoch-20-step-327-2017-06-19-18-13.ckpt-327 --image=./dataset/test/set5/mat --scale=8

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

from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.misc import imsave

import tensorflow as tf

from src.model import LapSRN_v1
from src.utils import sess_configure, trainsform, transform_reverse

from src.eval_dataset import eval_dataset
from src.evaluation import psnr as compute_psnr
from src.evaluation import _SSIMForMultiScale as compute_ssim

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="null", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="null", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=1, type=int, help="input image channel, Default: 4")
parser.add_argument("--sr_method", default="lapsrn", type=str, help="srn method")
parser.add_argument("--batch_size", default=2, type=int, help="batch size")

opt = parser.parse_args()

scale_list = [2, 4, 8]

def load_img(img_mat_path, scale):
  image_hash = sio.loadmat(img_mat_path)

  im_l_y = image_hash['label_x{}_y'.format(8//scale)]
  im_bicubic_ycbcr = image_hash['bicubic_l{}_x{}_ycbcr'.format(scale, scale)]
  im_bicubic_ycbcr = np.clip(im_bicubic_ycbcr*255., 0, 255.)
  img_gt = image_hash['label_x8_y']

  # im_l_y = image_hash['im_l_y']
  # im_l_ycbcr = image_hash['im_l_ycbcr']
  # im_bicubic_ycbcr = imresize(im_l_ycbcr, 4.0, interp='bicubic')
  # img_gt = image_hash['im_gt_y']
  # im_l_y = im_l_y/255.
  # img_gt = img_gt/255.

  # img_name = os.path.basename(img_mat_path).split('.')[0]
  # dir = os.path.dirname(img_mat_path)
  # im_l_ycbcr = imread(os.path.join(dir, '../lr_x2348', '{}_l{}.png'.format(img_name, opt.scale)), mode='YCbCr')
  # im_bicubic_ycbcr = imresize(im_l_ycbcr, 4.0, interp='bicubic')
  # im_l_ycbcr = im_l_ycbcr/255.
  # im_l_y = im_l_ycbcr[:,:,0]
  # img_gt = imread(os.path.join(dir, '../PNG', img_name+'.png'), mode='YCbCr')
  # img_gt = img_gt[:,:,0]/255.

  return im_l_y, im_bicubic_ycbcr, img_gt

def restore_img(im_h_y, im_h_ycbcr):
  im_h_y = np.clip(im_h_y*255., 0, 255.)

  img = np.zeros((im_h_y.shape[0], im_h_y.shape[1], 3), np.uint8)
  img[:,:,0] = im_h_y[:,:,0]
  img[:,:,1] = im_h_ycbcr[:,:,1]
  img[:,:,2] = im_h_ycbcr[:,:,2]
  img = Image.fromarray(img, "YCbCr").convert("RGB")

  return img

def generator(input_img, batch_size, scale, channel, model_path, gpu_id):

  graph = tf.Graph()
  sess_conf = sess_configure()

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

      model = LapSRN_v1(inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size=img_size, upscale_factor=scale, is_training=is_training)
      model.extract_features()
      model.reconstruct()
      upscaled_tf_img = model.upscaled_img(scale)

      saver = tf.train.Saver()
      if os.path.isdir(model_path):
        latest_ckpt = tf.train.latest_checkpoint(model_path)
        if latest_ckpt:
          saver.restore(sess, latest_ckpt)
          print("restore model from %s"%latest_ckpt)
        else:
          print("cannot restore model from %s, plz checkout"%latest_ckpt)
      else:
        saver.restore(sess, model_path)
        print("restore model from %s"%model_path)

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

def SR(dataset_dir, batch_size, init_scale, channel, model_path, gpu_id):

  dataset_image_path = os.path.join(dataset_dir, '*.mat')

  PSNR = []
  SSIM = []
  EXEC_TIME = []

  for scale in scale_list[0:np.log2(init_scale).astype(int)]:
    ssims = []
    psnrs = []
    exec_time = []

    for filepath in glob(dataset_image_path):

      im_l_y, im_h_ycbcr, img_gt_y = load_img(filepath, scale)
      im_h_y, elapsed_time = generator(im_l_y, batch_size, scale, channel, model_path, gpu_id)
      psnr, ssim = cal_image_index(img_gt_y, im_h_y[:,:,0])

      psnrs.append(psnr)
      ssims.append(ssim)
      exec_time.append(elapsed_time)

    PSNR.append(psnrs)
    SSIM.append(ssims)
    EXEC_TIME.append(exec_time)

  return PSNR, SSIM, EXEC_TIME


if __name__ == '__main__':

  if not os.path.exists(opt.output_dir) and opt.output_dir != 'null':
    os.system('mkdir -p ' + opt.output_dir)

  if os.path.isdir(opt.image):


    PSNR, SSIM, EXEC_TIME = SR(opt.image, opt.batch_size, opt.scale, opt.channel, opt.model, opt.gpu_id)

    for scale in scale_list[0:np.log2(opt.scale).astype(int)]:
      l = np.log2(scale).astype(int) - 1
      print("for dataset %s, scale: %d, exec time: %.4fs\n--PSNR: %.4f;\tSSIM: %.4f\n"%(opt.image, scale, np.mean(EXEC_TIME[l]), np.mean(PSNR[l]), np.mean(SSIM[l])));

  else:
    print("please set correct input")

