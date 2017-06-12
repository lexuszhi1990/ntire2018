#!/usr/bin/python
'''
usage:
single image:
  python test_ml.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-50-step-12-2017-06-10-23-03.ckpt-12 --image=./Set5/baby_GT.mat --gt_image=./dataset/test/set14/GT/baboon.png --output_dir=./dataset/test/set14/lapsrn/v1
for dataset:
  python test_misc.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-50-step-12-2017-06-10-23-03.ckpt-12 --image=./dataset/test/set14/lr_x2348 --output_dir=./dataset/test/set14/lapsrn/v15
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

from src.model import LapSRN
from src.model_tf import LapSRN as LapSRN_v1
from src.utils import sess_configure, trainsform, transform_reverse

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=1, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="./null", type=str, help="image path or single image")
parser.add_argument("--gt_image", default="", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="./null", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=1, type=int, help="input image channel, Default: 4")

opt = parser.parse_args()
batch_size = 2
sr_method = 'lapsrn'

def im2double(im):
  info = np.iinfo(im.dtype) # Get the data type of the input image
  return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def load_img_with_expand_dims(img_path, channel):
  img = imread(img_path, mode='YCbCr')
  height, width, _ = img.shape
  img = im2double(img)

  inputs = np.zeros((batch_size, height, width, channel))
  inputs[0] = img[:,:,0:channel]

  return inputs, (height, width)

def val_img_path(img_path, upscale_factor,output_dir=None):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_{}_x{}.png".format(img_name, sr_method, str(upscale_factor))
  if output_dir != None and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name)

def save_img(image, path):

  imsave(path, image)

  print("upscaled image size {}".format(np.shape(image)))
  print("save image at {}\n".format(path))

def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def restore_img(img_path, im_h_y):
  # img_h_y = np.clip(img_h_y*255., 0, 255.)
  im_h_y = im_h_y*255.
  im_h_y[im_h_y<0] = 0
  im_h_y[im_h_y>255.] = 255.
  im_h_y = im_h_y[:,:, 0]

  img = imread(img_path, mode='YCbCr')
  upscaled_im = imresize(img, 4.0, interp='bilinear')
  img_h = colorize(im_h_y, upscaled_im)

  return img_h

def generator(input_img):

  graph = tf.Graph()
  sess_conf = sess_configure()

  img_size = input_img.shape
  height, width = input_img.shape
  batch_images = np.zeros((batch_size, height, width, opt.channel))
  batch_images[0, :, :, 0] = input_img

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      gt_img_x2 = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      gt_img_x4 = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN_v1(inputs, gt_img_x2, gt_img_x4, image_size=img_size, upscale_factor=opt.scale, is_training=is_training)
      model.extract_features()
      model.reconstruct()
      upscaled_x4_img = model.sr_imgs[np.log2(opt.scale).astype(int)-1]

      saver = tf.train.Saver()
      if os.path.isdir(opt.model):
        latest_ckpt = tf.train.latest_checkpoint(opt.model)
        if latest_ckpt:
          saver.restore(sess, latest_ckpt)
          print("restore model from %s"%latest_ckpt)
        else:
          print("cannot restore model from %s, plz checkout"%latest_ckpt)
      else:
        saver.restore(sess, opt.model)
        print("restore model from %s"%opt.model)

      start_time = time.time()
      upscaled_img = sess.run(upscaled_x4_img, feed_dict={inputs: batch_images, is_training: False})
      elapsed_time = time.time() - start_time
      print("It takes {}s for processing\n".format(elapsed_time))

      return upscaled_img[0]

if __name__ == '__main__':

  if os.path.exists(opt.output_dir):
    os.system('rm -rf ' + opt.output_dir)
    os.mkdir(opt.output_dir)
  else:
    os.mkdir(opt.output_dir)

  if os.path.isdir(opt.image):

    dataset_image_path = os.path.join(opt.image, '*l{}.png'.format(opt.scale))
    for filepath in glob(dataset_image_path):
      print("upscale image: %s"%filepath)

      upscaled_img = generator(filepath)

      restored_img = restore_img(filepath, upscaled_img)
      output_img_path = val_img_path(filepath, opt.scale, opt.output_dir)
      save_img(restored_img, output_img_path)

  elif os.path.isfile(opt.image):

    im_l_ycbcr = sio.loadmat(opt.image)['label_x2_ycbcr']
    im_l_y = sio.loadmat(opt.image)['label_x2_y']
    im_h_ycbcr = imresize(im_l_ycbcr, 4.0, interp='bicubic')

    im_h_y = generator(im_l_y)
    im_h_y = np.clip(im_h_y*255., 0, 255.)

    img = np.zeros((im_h_y.shape[0], im_h_y.shape[1], 3), np.uint8)
    img[:,:,0] = im_h_y[:,:,0]
    img[:,:,1] = im_h_ycbcr[:,:,1]
    img[:,:,2] = im_h_ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")

    img_name = os.path.basename(opt.image).split('.')[0]
    save_img(img, img_name+'_v2.png')

  else:
    print("please set correct input")

