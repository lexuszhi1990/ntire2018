#!/usr/bin/python
'''
usage:
single image:
  python test_ml.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-50-step-24-2017-06-12-15-26.ckpt-24 --image=./dataset/test/set5/mat/baby_GT.mat --output_dir=./
for dataset:
  python test_ml.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-100-step-24-2017-06-12-15-36.ckpt-24 --image=./dataset/test/set5/mat --output_dir=./dataset/test/set5/lapsrn/v2
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
parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="null", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="null", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=1, type=int, help="input image channel, Default: 4")
parser.add_argument("--sr_method", default="lapsrn", type=str, help="srn method")
parser.add_argument("--batch_size", default=2, type=int, help="batch size")

opt = parser.parse_args()

def im2double(im):
  info = np.iinfo(im.dtype) # Get the data type of the input image
  return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def load_img(img_mat_path):
  image_hash = sio.loadmat(img_mat_path)
  im_l_ycbcr = image_hash['label_x{}_ycbcr'.format(8//opt.scale)]
  im_l_y = image_hash['label_x{}_y'.format(8//opt.scale)]
  im_h_ycbcr = imresize(im_l_ycbcr, 4.0, interp='bicubic')

  return im_l_y, im_h_ycbcr

def val_img_path(img_path):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_l{}_{}_x{}.png".format(img_name, opt.scale, opt.sr_method, str(opt.scale))
  if opt.output_dir != 'null' and os.path.isdir(opt.output_dir):
    dir = opt.output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name)

def save_img(image, path):
  output_img_path = val_img_path(path)
  imsave(output_img_path, image)

  print("upscaled image size {}".format(np.shape(image)))
  print("save image at {}".format(output_img_path))


def restore_img(im_h_y, im_h_ycbcr):
  im_h_y = np.clip(im_h_y*255., 0, 255.)

  img = np.zeros((im_h_y.shape[0], im_h_y.shape[1], 3), np.uint8)
  img[:,:,0] = im_h_y[:,:,0]
  img[:,:,1] = im_h_ycbcr[:,:,1]
  img[:,:,2] = im_h_ycbcr[:,:,2]
  img = Image.fromarray(img, "YCbCr").convert("RGB")

  return img

def generator(input_img):

  graph = tf.Graph()
  sess_conf = sess_configure()

  img_size = input_img.shape
  height, width = input_img.shape
  batch_images = np.zeros((opt.batch_size, height, width, opt.channel))
  batch_images[0, :, :, 0] = input_img

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [opt.batch_size, None, None, opt.channel])
      gt_img_x2 = tf.placeholder(tf.float32, [opt.batch_size, None, None, opt.channel])
      gt_img_x4 = tf.placeholder(tf.float32, [opt.batch_size, None, None, opt.channel])
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
      print("\nIt takes {}s for processing".format(elapsed_time))

      return upscaled_img[0]

def SR(input_mat_img):
  im_l_y, im_h_ycbcr = load_img(input_mat_img)
  im_h_y = generator(im_l_y)
  upscaled_img = restore_img(im_h_y, im_h_ycbcr)
  save_img(upscaled_img, input_mat_img)

if __name__ == '__main__':

  if not os.path.exists(opt.output_dir) and opt.output_dir != 'null':
    os.system('mkdir -p ' + opt.output_dir)

  if os.path.isdir(opt.image):
    dataset_image_path = os.path.join(opt.image, '*.mat')
    for filepath in glob(dataset_image_path):
      SR(filepath)

  elif os.path.isfile(opt.image):
    SR(opt.image)

  else:
    print("please set correct input")

