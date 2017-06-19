#!/usr/bin/python
'''
  usage: from src.eval_dataset import eval_dataset
        eval_dataset('./dataset/test/set14', 'bicubic', 'bicubic', 4)
'''

import os
from glob import glob
import cv2
import numpy as np

from PIL import Image
from scipy.ndimage import imread

from src.evaluation import psnr as compute_psnr
from src.evaluation import _SSIMForMultiScale as compute_ssim

def preprocess(img_path, shave_bd=0):
  img = imread(img_path, mode='YCbCr')
  height, width = img.shape[:2]
  img = img[shave_bd:height - shave_bd, shave_bd:width - shave_bd]

  return img

def eval_dataset(dataset_dir, test_dir, sr_method, scale):
  gt_img_dir = os.path.join(dataset_dir, 'PNG')
  gt_img_list = glob(os.path.join(gt_img_dir, '*.*'))

  PSNR = []
  SSIM = []

  for image_ab_path in gt_img_list:

    image_basename = os.path.basename(image_ab_path).split('.')[0]
    test_img_name = '{}_l{}_{}_x{}.png'.format(image_basename, scale, sr_method, scale)
    upscaled_img_path = os.path.join(dataset_dir, test_dir, test_img_name)

    gt_img = preprocess(image_ab_path, shave_bd=0)
    upscaled_img = preprocess(upscaled_img_path, shave_bd=0)

    gt_img_y = gt_img[:,:,0]
    upscaled_img_y = upscaled_img[:,:,0]

    psnr = compute_psnr(gt_img_y, upscaled_img_y)
    PSNR.append(psnr)

    gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
    upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
    ssim = compute_ssim(gt_img_ep, upscaled_img_ep)[0]
    SSIM.append(ssim)

    print("for image: %s\n--PSNR: %.4f;\tSSIM: %.4f"%(upscaled_img_path, psnr, ssim));

  print("\nfor saved image %s:\n--PSNR: %.4f;\tSSIM: %.4f"%(test_dir, np.mean(PSNR), np.mean(SSIM)));

  return PSNR, SSIM



