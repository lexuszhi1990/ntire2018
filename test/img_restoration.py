# -*- coding: utf-8 -*-

import time
import argparse
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matlab
import matlab.engine
from src.cv2_utils import *
from src.utils import sess_configure, trainsform, transform_reverse

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19

def save_mat(img, path, sr_method, scale):
  image_hash = sio.loadmat(path)
  img_key = '{}_l{}_x{}_y'.format(sr_method, scale, scale)
  image_hash[img_key] = img
  sio.savemat(path, image_hash)
  print('save mat at {} in {}'.format(path, img_key))

def val_img_path(img_path, scale, output_dir):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_sr_x{}.png".format(img_name, scale)
  upscaled_mat_name =  "{}_sr_x{}.mat".format(img_name, scale)
  bicubic_img_name =  "{}_bicubic_x{}.png".format(img_name, scale)
  if output_dir != 'null' and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name), os.path.join(dir, upscaled_mat_name), os.path.join(dir, bicubic_img_name)


img_path = '0801x4d.png'
scale = 4
output_dir = '.'

saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, output_dir)

eng = matlab.engine.start_matlab()
eng.addpath(r'./src/evaluation_mat', nargout=0)
eng.addpath(r'./src/evaluation_mat/ifc-drrn')
eng.addpath(r'./src/evaluation_mat/matlabPyrTools')

aa = eng.get_ycbcr_image(img_path, mat_dir, scale);

image_hash = sio.loadmat(mat_dir)
img_y = image_hash['img_y']
save_mat(hr_img, mat_dir)
eng.save_ycbcr_image(mat_dir, saved_dir, mat_dir)

### split the image and train
input_img = img_y
batch_size = 1
scale = 4
channel = 1
filter_num = 64
gpu_id = 3
pad = 2
model_name = 'LapSRN_v7'
model_path ='./saved_models/x4/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774'

from val import generator
generator(img_1, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)















def generator(input_img, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id):

height, width = input_img.shape
hr_img = np.zeros((height*4, width*4))

img_1 = input_img[:height/3+pad,:width/3+pad]
batch_images = np.zeros((batch_size, img_1.shape[0], img_1.shape[1], channel))
batch_images[0, :, :, 0] = img_1
img_size = img_1.shape
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
  upscaled_tf_img = model.upscaled_img(scale)

sess_conf = sess_configure(memory_per=.95)
sess = tf.Session(config=sess_conf)
saver = tf.train.Saver()
saver.restore(sess, model_path)
upscaled_img = sess.run(upscaled_tf_img, feed_dict={inputs: batch_images, is_training: False})

upscaed_img_1 = upscaled_img[0][:upscaled_img[0].shape[0]-pad*4, :upscaled_img[0].shape[1]-pad*4, 0]
hr_img[:height/3*4,:width/3*4] = upscaed_img_1

"""
>>> img_1.shape
(115, 172)
>>> upscaed_img_1.shape
(452, 680)
"""

img_2 = input_img[:height/3,width/3:width/3*2]
img_3 = input_img[:height/3,width/3*2:]
img_4 = input_img[height/3:height/3*2,:width/3]
img_5 = input_img[height/3:height/3*2,width/3:width/3*2]
img_6 = input_img[height/3:height/3*2,width/3*2:]
img_7 = input_img[height/3*2:,:width/3]
img_8 = input_img[height/3*2:,width/3:width/3*2]
img_9 = input_img[height/3*2:,width/3*2:]



input_img = img_y
batch_size = 1
scale = 4
channel = 1
filter_num = 64
gpu_id = 3
pad = 2
sr_method= 'edsr'
model_name = 'LapSRN_v7'
model_path ='./saved_models/x4/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774'
output_dir = '.'

img_path = '0801x4d.png'


saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, sr_method, output_dir=None, verbose=True)
eng.get_ycbcr_image(img_path, mat_dir, scale);
image_hash = sio.loadmat(mat_dir)
input_img = image_hash['img_y']

height, width = input_img.shape
hr_img = np.zeros((height*4, width*4))

img_1 = input_img[:height/3+pad,:width/3+pad]
upscaled_img = generator(img_1, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_1 = upscaled_img[0][:upscaled_img[0].shape[0]-pad*4, :upscaled_img[0].shape[1]-pad*4, 0]
hr_img[:height/3*4,:width/3*4] = upscaed_img_1

img_2 = input_img[:height/3+2,width/3-1:width/3*2+1]
upscaled_img = generator(img_2, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_2 = upscaled_img[0][:upscaled_height-pad*4, 4:upscaled_width-4, 0]
hr_img[:height/3*4,width/3*4:width/3*2*4] = upscaed_img_2

img_3 = input_img[:height/3,width/3*2:]
upscaled_img = generator(img_3, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_3 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[:height/3*4,width/3*2*4:] = upscaed_img_3

img_4 = input_img[height/3:height/3*2,:width/3]
upscaled_img = generator(img_4, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_4 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,:width/3*4] = upscaed_img_4

img_5 = input_img[height/3:height/3*2,width/3:width/3*2]
upscaled_img = generator(img_5, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_5 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*4:width/3*2*4] = upscaed_img_5

img_6 = input_img[height/3:height/3*2,width/3*2:]
upscaled_img = generator(img_6, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_6 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*2*4:] = upscaed_img_6

img_7 = input_img[height/3*2:,:width/3]
upscaled_img = generator(img_7, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_7 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,:width/3*4] = upscaed_img_7

img_8 = input_img[height/3*2:,width/3:width/3*2]
upscaled_img = generator(img_8, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_8 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*4:width/3*2*4] = upscaed_img_8

img_9 = input_img[height/3*2:,width/3*2:]
upscaled_img = generator(img_9, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_9 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*2*4:] = upscaed_img_9


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



img_path = '0801x4d.png'
scale = 4
output_dir = '.'

saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, output_dir)

null = eng.get_ycbcr_image(img_path, mat_dir, scale);
image_hash = sio.loadmat(mat_dir)
img_y = image_hash['img_y']
save_mat(hr_img, mat_dir)
null = eng.save_ycbcr_image(mat_dir, saved_dir, bicubic_dir)



input_img = img_y
batch_size = 1
scale = 4
channel = 1
filter_num = 64
gpu_id = 3
pad = 2
sr_method= 'edsr'
model_name = 'LapSRN_v7'
model_path ='./saved_models/x4/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774'
output_dir = '.'

img_path = '0801x4d.png'


saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, sr_method, output_dir=None, verbose=True)
eng.get_ycbcr_image(img_path, mat_dir, scale);
image_hash = sio.loadmat(mat_dir)
input_img = image_hash['img_y']

height, width = input_img.shape
hr_img = np.zeros((height*4, width*4))

img_1 = input_img[:height/3+pad,:width/3+pad]
upscaled_img = generator(img_1, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_1 = upscaled_img[0][:upscaled_img[0].shape[0]-pad*4, :upscaled_img[0].shape[1]-pad*4, 0]
hr_img[:height/3*4,:width/3*4] = upscaed_img_1

img_2 = input_img[:height/3+2,width/3-1:width/3*2+1]
upscaled_img = generator(img_2, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_2 = upscaled_img[0][:upscaled_height-pad*4, 4:upscaled_width-4, 0]
hr_img[:height/3*4,width/3*4:width/3*2*4] = upscaed_img_2

img_3 = input_img[:height/3,width/3*2:]
upscaled_img = generator(img_3, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_3 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[:height/3*4,width/3*2*4:] = upscaed_img_3

img_4 = input_img[height/3:height/3*2,:width/3]
upscaled_img = generator(img_4, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_4 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,:width/3*4] = upscaed_img_4

img_5 = input_img[height/3:height/3*2,width/3:width/3*2]
upscaled_img = generator(img_5, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_5 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*4:width/3*2*4] = upscaed_img_5

img_6 = input_img[height/3:height/3*2,width/3*2:]
upscaled_img = generator(img_6, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_6 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*2*4:] = upscaed_img_6

img_7 = input_img[height/3*2:,:width/3]
upscaled_img = generator(img_7, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_7 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,:width/3*4] = upscaed_img_7

img_8 = input_img[height/3*2:,width/3:width/3*2]
upscaled_img = generator(img_8, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_8 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*4:width/3*2*4] = upscaed_img_8

img_9 = input_img[height/3*2:,width/3*2:]
upscaled_img = generator(img_9, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_9 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*2*4:] = upscaed_img_9
