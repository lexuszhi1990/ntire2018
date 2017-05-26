from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import cv2
import os
import numpy as np

import tensorflow as tf

from src.model import LapSRN
from src.utils import sess_configure
from src.evalution import psnr as compute_psnr
from src.evalution import _SSIMForMultiScale as compute_ssim

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=1, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="./dataset", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="./dataset", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="input image channel, Default: 4")

# usage:
# single image:
#   python test.py --gpu_id=1 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./tmp/test_imgs/a1.jpg --scale=4 --channel=3
# for dataset:
#   python test.py --gpu_id=1 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./dataset/test/set5/GT --output_dir=./dataset/test/set5/lapsrn --scale=4 --channel=3

opt = parser.parse_args()
batch_size = 2

def load_img_with_expand_dims(img_path):
  img = cv2.imread(img_path, opt.channel)
  img = np.array(img)/127.5 - 1.
  height, width, _ = img.shape
  inputs = np.zeros((batch_size, height, width, opt.channel))
  inputs[0] = img

  return inputs, (height, width)

def upscaled_img_path(img_path, upscale_factor,output_dir=None):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_lapsrn_x{}.png".format(img_name, str(upscale_factor))
  if output_dir != None and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name)

def transform_reverse(images):
  images = (images + 1.) * 127.5
  # images[images<0] = 0
  # images[images>255.] = 255.

  return np.clip(images, 0, 255.)

def generator(input_img, output_path):

  if not tf.gfile.Exists(input_img):
    print('\nCannot find .\n')
    return

  batch_images, img_size = load_img_with_expand_dims(input_img)

  graph = tf.Graph()
  sess_conf = sess_configure()

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      gt_imgs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN(inputs, gt_imgs, image_size=img_size, upscale_factor=opt.scale, is_training=is_training)
      model.extract_features()
      model.reconstruct()

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
      upscaled_img = sess.run(model.sr_imgs[-1], feed_dict={inputs: batch_images, is_training: False})
      elapsed_time = time.time() - start_time

      start_time = time.time()
      upscaled_img = sess.run(model.sr_imgs[-1], feed_dict={inputs: batch_images, is_training: False})
      elapsed_time = time.time() - start_time

      transformed_img = transform_reverse(upscaled_img)
      cv2.imwrite(output_path, transformed_img[0])

      print("It takes {}s for processing\n".format(elapsed_time))
      print("save image at {}\n".format(output_path))

if __name__ == '__main__':

  if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

  if os.path.isdir(opt.image):
    list = os.listdir(opt.image)
    for image in list:
      filepath = os.path.join(opt.image, image)
      if os.path.isfile(filepath):
        print(filepath)
        output_img_path = upscaled_img_path(filepath, opt.scale, opt.output_dir)
        generator(filepath, output_img_path)

  elif os.path.isfile(opt.image):
    output_img_path = upscaled_img_path(opt.image, opt.scale, opt.output_dir)
    generator(opt.image, output_img_path)
  else:
    print("please set correct input")

