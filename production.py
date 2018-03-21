#!/usr/bin/python
'''
'''

import time
import argparse
import os
import numpy as np
from glob import glob
import scipy.io as sio
import shutil
import tensorflow as tf
import matlab
import matlab.engine

from src.utils import sess_configure, trainsform, transform_reverse
from src.eval_dataset import eval_dataset
from src.evaluation import shave_bd, compute_psnr, compute_ssim, compute_msssim

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19
from src.model_new import EDSR_v301, EDSR_v302, EDSR_v303, EDSR_v304, EDSR_v305, EDSR_v306, EDSR_v307, EDSR_v308, EDSR_v309, EDSR_v310, EDSR_v311, EDSR_v312, EDSR_v313, EDSR_v314, EDSR_v315, EDSR_v316, EDSR_v321, EDSR_v322, EDSR_v323, EDSR_v324, EDSR_v325, EDSR_v326, EDSR_v327, EDSR_v328, EDSR_v329, EDSR_v330

eng = matlab.engine.start_matlab()
eng.addpath(r'./src/evaluation_mat', nargout=0)
eng.addpath(r'./src/dataset_builder', nargout=0)

def stop_matlab():
  eng.quit()

def val_img_path(img_path, scale, sr_method, output_dir=None, verbose=False):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_l{}_{}_x{}.png".format(img_name, scale, sr_method, str(scale))
  upscaled_mat_name =  "{}_sr_x{}.mat".format(img_name, scale)
  bicubic_img_name =  "{}_bicubic_x{}.png".format(img_name, scale)

  if output_dir is not None and os.path.isdir(output_dir):
    base_dir = output_dir
  else:
    base_dir = os.path.dirname(img_path)

  if verbose is False:
    return os.path.join(base_dir, upscaled_img_name)
  else:
    return os.path.join(base_dir, upscaled_img_name), os.path.join(base_dir, upscaled_mat_name), os.path.join(base_dir, bicubic_img_name)

def get_save_path(img_path, scale, sr_method, output_dir=None, verbose=False):
  img_base_name = os.path.basename(img_path).split('.')[0]
  upscaled_mat_name =  "{}.mat".format(img_base_name, scale)
  bicubic_img_name =  "{}_bicubic_x{}.png".format(img_base_name, scale)

  if output_dir is not None and os.path.isdir(output_dir):
    base_dir = output_dir
  else:
    base_dir = os.path.dirname(img_path)

  if verbose is False:
    return os.path.join(base_dir, os.path.basename(img_path))
  else:
    return os.path.join(base_dir, os.path.basename(img_path)), os.path.join(base_dir, upscaled_mat_name), os.path.join(base_dir, bicubic_img_name)

def save_img(image, img_path, scale, sr_method, output_dir):
  output_img_path = val_img_path(img_path)
  imsave(output_img_path, image)
  print("upscaled image size {}".format(np.shape(image)))
  print("save image at {}".format(output_img_path))

def save_mat(img, path, sr_method='edsr', scale=4):
  image_hash = sio.loadmat(path)
  # img_key = '{}_l{}_x{}_y'.format(sr_method, scale, scale)
  img_key = 'sr_img_y'
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

def build_image(input_img, model_path, model_name, batch_size, scale, channel, filter_num, sr_method, gpu_id):

  height, width = input_img.shape
  hr_img = np.zeros((height*4, width*4))

  img_1 = input_img[:height/3+2,:width/3+2]
  upscaled_img = generator(img_1, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_1 = upscaled_img[0][:upscaled_img[0].shape[0]-2*scale, :upscaled_img[0].shape[1]-2*scale, 0]
  hr_img[:height/3*scale,:width/3*scale] = upscaed_img_1

  img_2 = input_img[:height/3+2,width/3-1:width/3*2+1]
  upscaled_img = generator(img_2, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_2 = upscaled_img[0][:upscaled_height-2*scale, scale:upscaled_width-scale, 0]
  hr_img[:height/3*scale,width/3*scale:width/3*2*scale] = upscaed_img_2

  img_3 = input_img[:height/3+2,width/3*2-2:]
  upscaled_img = generator(img_3, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_3 = upscaled_img[0][:upscaled_height-2*scale, 2*scale:, 0]
  hr_img[:height/3*scale,width/3*2*scale:] = upscaed_img_3

  img_4 = input_img[height/3-1:height/3*2+1,:width/3+2]
  upscaled_img = generator(img_4, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_4 = upscaled_img[0][scale:upscaled_height-scale, :upscaled_width-2*scale, 0]
  hr_img[height/3*scale:height/3*2*scale,:width/3*scale] = upscaed_img_4

  img_5 = input_img[height/3-1:height/3*2+1,width/3-1:width/3*2+1]
  upscaled_img = generator(img_5, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_5 = upscaled_img[0][scale:upscaled_height-scale, scale:upscaled_width-scale, 0]
  hr_img[height/3*scale:height/3*2*scale,width/3*scale:width/3*2*scale] = upscaed_img_5

  img_6 = input_img[height/3-1:height/3*2+1,width/3*2-2:]
  upscaled_img = generator(img_6, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_6 = upscaled_img[0][scale:upscaled_height-scale, scale*2:, 0]
  hr_img[height/3*scale:height/3*2*scale,width/3*2*scale:] = upscaed_img_6

  img_7 = input_img[height/3*2-2:,:width/3+2]
  upscaled_img = generator(img_7, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_7 = upscaled_img[0][scale*2:upscaled_height, :upscaled_width-scale*2, 0]
  hr_img[height/3*2*scale:,:width/3*scale] = upscaed_img_7

  img_8 = input_img[height/3*2-2:,width/3-1:width/3*2+1]
  upscaled_img = generator(img_8, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_8 = upscaled_img[0][scale*2:upscaled_height, scale:upscaled_width-scale, 0]
  hr_img[height/3*2*scale:,width/3*scale:width/3*2*scale] = upscaed_img_8

  img_9 = input_img[height/3*2-2:,width/3*2-2:]
  upscaled_img = generator(img_9, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
  upscaled_height, upscaled_width, _ = upscaled_img[0].shape
  upscaed_img_9 = upscaled_img[0][2*scale:upscaled_height, 2*scale:upscaled_width, 0]
  hr_img[height/3*2*scale:,width/3*2*scale:] = upscaed_img_9

  return hr_img

def SR(dataset_dir, model_path, model_name, output_dir=None,gpu_id=3, batch_size=1, scale=4, channel=1, filter_num=64, sr_method='edsr'):

  dataset_image_path = os.path.join(dataset_dir, '*.png')
  for filepath in glob(dataset_image_path):

    saved_dir, mat_dir, bicubic_dir = get_save_path(filepath, scale, sr_method, output_dir=output_dir, verbose=True)
    null = eng.get_ycbcr_image(filepath, mat_dir, scale);
    image_hash = sio.loadmat(mat_dir)
    input_img = image_hash['img_y']

    im_h_y = build_image(input_img, model_path, model_name, batch_size, scale, channel, filter_num, sr_method, gpu_id)

    save_mat(im_h_y, mat_dir)
    null = eng.save_ycbcr_image(mat_dir, saved_dir, bicubic_dir)


if __name__ == '__main__':
  # v1:
  # model_path ='./saved_models/x4/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774'
  # model_name = 'LapSRN_v7'
  # img_dir = '../dataset/DIV2K_valid_LR_difficult'
  # output_dir = './'

  # v2:
  model_path ='./ckpt/k40_EDSR313/EDSR_v313-epoch-1-step-10000-2018-03-21-20-03.ckpt-10000'
  model_name = 'EDSR_v313'
  img_dir = '../dataset/DIV2K_valid_LR_difficult'
  output_dir = '../dataset/VALIDATION/res_v2'

  SR(img_dir, model_path, model_name, output_dir=output_dir, gpu_id=3)

  stop_matlab()
