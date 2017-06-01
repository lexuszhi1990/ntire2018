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
parser.add_argument("--output_dir", default="./dataset", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="input image channel, Default: 4")

# usage:
# single image:
#   python test.py --gpu_id=3 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./tmp/test_imgs/a1.jpg --scale=4 --channel=3
# for dataset:
#   python test.py --gpu_id=3 --channel=3 --model=ckpt/lapsrn --image=./dataset/test/set14/lr_x2348 --output_dir=./dataset/test/set14/lapsrn_v1 --scale=4
#   python test.py --gpu_id=3 --channel=3 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./dataset/test/set14/lr_x2348 --output_dir=./dataset/test/set14/lapsrn_v1 --scale=4
#   python test.py --gpu_id=3 --channel=1 --scale=4 --model=./ckpt/lapsrn/lapsrn-epoch-100-step-35-17-06-01-16-22.ckpt-35 --image=./dataset/test/set14/lr_x2348/baboon_l4.png --output_dir=./dataset/test/set14/lapsrn/v1

opt = parser.parse_args()
batch_size = 2
sr_method = 'lapsrn'

def im2double(im):
  info = np.iinfo(im.dtype) # Get the data type of the input image
  return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def load_img_with_expand_dims(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
  img = im2double(img)
  height, width, _ = img.shape

  inputs = np.zeros((batch_size, height, width, opt.channel))
  inputs[0] = img[:,:,0:opt.channel]

  return inputs, (height, width)

def upscaled_img_path(img_path, upscale_factor,output_dir=None):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_{}_x{}.png".format(img_name, sr_method, str(upscale_factor))
  if output_dir != None and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name)

def generator(graph, sess_conf, input_img, output_path, reuse=False):

  if not tf.gfile.Exists(input_img):
    print('\nCannot find .\n')
    return

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      gt_imgs = tf.placeholder(tf.float32, [batch_size, None, None, opt.channel])
      is_training = tf.placeholder(tf.bool, [])

      batch_images, img_size = load_img_with_expand_dims(input_img)

      model = LapSRN(inputs, gt_imgs, image_size=img_size, upscale_factor=opt.scale, is_training=is_training)
      model.extract_features(reuse=reuse)
      model.reconstruct(reuse=reuse)
      upscaled_x4_img = transform_reverse(model.sr_imgs[1])

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

      scipy.misc.imsave(output_path, upscaled_img)
      print("save image at {}\n".format(output_path))

      sess.close()

if __name__ == '__main__':

  graph = tf.Graph()
  sess_conf = sess_configure()

  if os.path.exists(opt.output_dir):
    os.system('rm -rf ' + opt.output_dir)
    os.mkdir(opt.output_dir)
  else:
    os.mkdir(opt.output_dir)

  if os.path.isdir(opt.image):
    image_path = os.path.join(opt.image, '*l{}.png'.format(opt.scale))
    list = glob(image_path)
    for filepath in list:
      print("upscale image: %s"%filepath)
      if os.path.isfile(filepath):
        output_img_path = upscaled_img_path(filepath, opt.scale, opt.output_dir)
        # if filepath != list[0]:
        #   tf.reset_default_graph()
        # generator(graph, sess_conf, filepath, output_img_path)
        generator(graph, sess_conf, filepath, output_img_path, filepath != list[0])

  elif os.path.isfile(opt.image):
    output_img_path = upscaled_img_path(opt.image, opt.scale, opt.output_dir)
    generator(graph, sess_conf, opt.image, output_img_path)

  else:
    print("please set correct input")

