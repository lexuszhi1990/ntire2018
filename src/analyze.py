import cv2
import os
import numpy as np

from src.evaluation import psnr as compute_psnr
from src.evaluation import _SSIMForMultiScale as compute_ssim

def compare_internal(gt_img_path, upscaled_img_path, patch_size=30, output_path='.'):

  gt_img = cv2.imread(gt_img_path)
  upscaled_img = cv2.imread(upscaled_img_path)

  gt_img_y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]
  upscaled_img_y = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]

  mean_psnr = compute_psnr(gt_img_y, upscaled_img_y)
  # mean_ssim = compute_ssim(gt_img_y, upscaled_img_y)

  bg_image = np.zeros(gt_img.shape, np.uint8)
  batch_height, batch_width = gt_img.shape[0]//patch_size, gt_img.shape[1]//patch_size

  for h in range(batch_height):
    for w in range(batch_width):

      gt_patch = gt_img_y[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
      upsacled_patch = upscaled_img_y[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]

      psnr = compute_psnr(gt_patch, upsacled_patch)
      if psnr > mean_psnr:
        bg_image[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size, 2] = 127
      else:
        bg_image[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size, 1] = 127

  merged_img = cv2.addWeighted(gt_img,0.5,bg_image,0.5,0)
  path = os.path.join(output_path, 'analyzed_{}_{}'.format(patch_size, os.path.basename(gt_img_path)))
  cv2.imwrite(path, merged_img)
  print('saved img at %s' % path)

if __name__ == '__main__':
  # from src.analyze import compare_internal
  gt_img = './100075.png'
  upscaled_img = './100075_l4_x4.png'
  compare_internal('./dataset/test/set14/PNG/baboon.png', './dataset/test/set14/lapsrn/v1-27.57/baboon_l4_lapsrn_x4.png', 10, './tmp')
