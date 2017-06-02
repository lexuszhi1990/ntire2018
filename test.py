#!/usr/bin/python
'''
usage:
single image:
  python test.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-100-step-35-17-06-01-16-22.ckpt-35 --image=./dataset/test/set14/lr_x2348/baboon_l4.png --gt_image=./dataset/test/set14/GT/baboon.png --output_dir=./dataset/test/set14/lapsrn/v1
for dataset:
  python test.py --gpu_id=3 --channel=3 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-50-step-27-2017-06-02-21-27.ckpt-27 --image=./dataset/test/set14/lr_x2348 --output_dir=./dataset/test/set14/lapsrn/v6
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import cv2
import scipy.misc
import os
import numpy as np
from glob import glob

import tensorflow as tf

from src.model import LapSRN
from src.utils import sess_configure, trainsform, transform_reverse

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=1, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="./dataset", type=str, help="image path or single image")
parser.add_argument("--gt_image", default="", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="./dataset", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="input image channel, Default: 4")

opt = parser.parse_args()
batch_size = 2
sr_method = 'lapsrn'

def im2double(im):
  info = np.iinfo(im.dtype) # Get the data type of the input image
  return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def load_img_with_expand_dims(img_path, channel):
  img = scipy.misc.imread(img_path)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img = im2double(img)
  height, width, _ = img.shape

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

  print("upscaled image size {}".format(np.shape(image)))

  # upscaled_rgb_img = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
  # upscaled_int_img = cv2.convertScaleAbs(upscaled_rgb_img, alpha=255)
  # cv2.imwrite(path, upscaled_int_img)

  # upscaled_rgb_img = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)
  # scipy.misc.imsave(path, upscaled_rgb_img)

  scipy.misc.imsave(path, image)

  print("save image at {}\n".format(path))

def generator(input_img):

  graph = tf.Graph()
  sess_conf = sess_configure()

  batch_images, img_size = load_img_with_expand_dims(input_img, opt.channel)

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      gt_imgs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN(inputs, gt_imgs, image_size=img_size, upscale_factor=opt.scale, is_training=is_training)
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

      output_img_path = val_img_path(filepath, opt.scale, opt.output_dir)
      save_img(upscaled_img, output_img_path)

  elif os.path.isfile(opt.image):
    output_img_path = val_img_path(opt.image, opt.scale, opt.output_dir)

  else:
    print("please set correct input")

