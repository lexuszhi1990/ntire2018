from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import cv2
import os

import tensorflow as tf

from src.model import LapSRN

parser = argparse.ArgumentParser(description="LapSRN Test")
parser.add_argument("--gpu_id", default=1, type=int, help="GPU id")
parser.add_argument("--model", default="ckpt/lapsrn", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--channel", default=3, type=int, help="input image channel, Default: 4")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")

# usage: python test.py --gpu_id=1 --model=ckpt/lapsrn/laprcn-model-17-05-25-15-59.ckpt-707 --image=./tmp/test_imgs/g22.png --scale=4 --channel=3

opt = parser.parse_args()

def load_img_with_expand_dims(img_path):
  img = cv2.imread(img_path, opt.channel)
  return np.expand_dims(img, axis=0)

def upscaled_img_path(img_path, upscale_factor):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_x{}.png".format(img_name, str(upscale_factor))
  return os.path.join(os.path.dirname(img_path), upscaled_img_name)

def generator():

  if not tf.gfile.Exists(opt.image):
    print('\nCannot find --original_image.\n')
    return

  img = load_img_with_expand_dims(opt.image)
  _, height, width, _ = img.shape

  graph = tf.Graph()
  sess_conf = sess_configure()

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(opt.gpu_id))):

      inputs = tf.placeholder(tf.float32, [1, None, None, opt.channel])
      gt_imgs = tf.placeholder(tf.float32, [1, None, None, opt.channel])
      is_training = tf.placeholder(tf.bool, [])

      model = LapSRN(inputs, gt_imgs, image_size=[height, width], upscale_factor=opt.scale, is_training=False)
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

      start_time = time.time()
      upscaled_img = sess.run(model.sr_imgs[-1], feed_dict={inputs: img, is_training: False})
      elapsed_time = time.time() - start_time

      cv2.imwrite(upscaled_img_path(opt.img, opt.scale), upscaled_img)
      print("It takes {}s for processing".format(elapsed_time))


if __name__ == '__main__':
  generator()
