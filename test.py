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

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=1, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="./dataset/test.png", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="input image channel, Default: 4")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")

# usage: python test.py --gpu_id=1 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./tmp/test_imgs/a1.jpg --scale=4 --channel=3

opt = parser.parse_args()
batch_size = 2

def load_img_with_expand_dims(img_path):
  img = cv2.imread(img_path, opt.channel)
  img = np.array(img)/127.5 - 1.
  _, height, width, _ = img.shape
  inputs = np.zeros((batch_size, height, width, opt.channel))
  inputs[0] = img

  return inputs, (height, width)

def upscaled_img_path(img_path, upscale_factor):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_lapsrn_x{}.png".format(img_name, str(upscale_factor))
  return os.path.join(os.path.dirname(img_path), upscaled_img_name)

def transform_reverse(images):
  images = (images+1.)/2. * 255.0
  images[images<0] = 0
  images[images>255.] = 255.

  return images

def generator():

  if not tf.gfile.Exists(opt.image):
    print('\nCannot find --original_image.\n')
    return

  batch_images, img_size = load_img_with_expand_dims(opt.image)
  print(img_size)

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

      print("It takes {}s for processing".format(elapsed_time))

      transformed_img = upscaled_img(upscaled_img)
      cv2.imwrite(upscaled_img_path(opt.image, opt.scale), transformed_img[0])

if __name__ == '__main__':
  generator()
