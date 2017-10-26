#!/usr/bin/python
'''
usage:
CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 --batch_size=1 --channel=1 --filter_num=64 --sr_method=EDSR_v241 --model_path=./ckpt/EDSR_v241/EDSR_v241-epoch-1-step-19548-2017-10-12-16-02.ckpt-19548 --image=./tmp/analyzed_10_man.png --output_dir=./ --scale=2

CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 --batch_size=1 --channel=1 --filter_num=64 --sr_method=EDSR_v250 --model_path=./ckpt/EDSR_v250/EDSR_v250-epoch-1-step-19548-2017-10-16-13-17.ckpt-19548 --image=./00001.jpg --output_dir=./ --scale=4

'''

import time
import argparse
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matlab
import matlab.engine

from src.model import LapSRN_v1, LapSRN_v2, LapSRN_v3, LapSRN_v4, LapSRN_v5, LapSRN_v6, LapSRN_v7, LapSRN_v8, LapSRN_v9, LapSRN_v10, LapSRN_v11, LapSRN_v12, LapSRN_v13, LapSRN_v14, LapSRN_v15, LapSRN_v16, LapSRN_v17, LapSRN_v18, LapSRN_v19
from src.model import LapSRN_v2_v1, LapSRN_v2_v2
from src.model import LapSRN_v30, LapSRN_v31, LapSRN_v32, LapSRN_v33, LapSRN_v34
from src.model import LapSRN_v40, LapSRN_v41, LapSRN_v42, LapSRN_v43, LapSRN_v44
from src.model_new import EDSR_v100, EDSR_v101, EDSR_v102, EDSR_v103, EDSR_v104, EDSR_v105, EDSR_v106
from src.model_new import EDSR_v201, EDSR_v202, EDSR_v203, EDSR_v204, EDSR_v205, EDSR_v206, EDSR_v207, EDSR_v208, EDSR_v209, EDSR_v210, EDSR_v211, EDSR_v212, EDSR_v213, EDSR_v214, EDSR_v215, EDSR_v216, EDSR_v217, EDSR_v218, EDSR_v219, EDSR_v220, EDSR_v221, EDSR_v222, EDSR_v223, EDSR_v224, EDSR_v225, EDSR_v226, EDSR_v227, EDSR_v228, EDSR_v229, EDSR_v230, EDSR_v241, EDSR_v242, EDSR_v243, EDSR_v244, EDSR_v245, EDSR_v246, EDSR_v247, EDSR_v248, EDSR_v249, EDSR_v250, EDSR_v251, EDSR_v252, EDSR_v253, EDSR_v254, EDSR_v255
from src.model_new import EDSR_v301, EDSR_v302, EDSR_v303, EDSR_v304, EDSR_v305, EDSR_v306, EDSR_v307, EDSR_v308, EDSR_v309, EDSR_v310, EDSR_v311, EDSR_v312, EDSR_v313, EDSR_v314, EDSR_v315, EDSR_v321, EDSR_v322, EDSR_v323, EDSR_v324, EDSR_v325, EDSR_v326, EDSR_v327, EDSR_v328, EDSR_v329, EDSR_v330
from src.model_new import EDSR_LFW_v1, EDSR_LFW_v2, EDSR_LFW_v3, EDSR_LFW_v4

from src.cv2_utils import *
from src.utils import sess_configure, trainsform, transform_reverse


# CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 --batch_size=1 --channel=1 --filter_num=64 --sr_method=EDSR_v241 --model_path=./ckpt/EDSR_v241/EDSR_v241-epoch-1-step-19548-2017-10-12-16-02.ckpt-19548 --output_dir=./ --image=./tmp/analyzed_10_man.png --scale=2

parser = argparse.ArgumentParser(description="EDSR Test")
parser.add_argument("--gpu_id", default=3, type=int, help="GPU id")
parser.add_argument("--sr_method", default="EDSR", type=str, help="model name")
parser.add_argument("--model_path", default="ckpt/edsr", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--filter_num", default=64, type=int, help="filter num")
parser.add_argument("--channel", default=1, type=int, help="conv channel")
parser.add_argument("--image", default="null", type=str, help="image path or single image")
parser.add_argument("--output_dir", default="null", type=str, help="image path")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")

opt = parser.parse_args()

def save_mat(img, path, sr_method, scale):
  image_hash = sio.loadmat(path)
  img_key = '{}_l{}_x{}_y'.format(sr_method, scale, scale)
  image_hash[img_key] = img
  sio.savemat(path, image_hash)

  print('save mat at {} in {}'.format(path, img_key))

def val_img_path(img_path, scale, output_dir):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_x{}.png".format(img_name, scale)
  upscaled_mat_name =  "{}_x{}.mat".format(img_name, scale)
  bicubic_img_name =  "{}_bicubic_x{}.png".format(img_name, scale)
  if output_dir != 'null' and os.path.isdir(output_dir):
    dir = output_dir
  else:
    dir = os.path.dirname(img_path)
  return os.path.join(dir, upscaled_img_name), os.path.join(dir, upscaled_mat_name), os.path.join(dir, bicubic_img_name)

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
      upscaled_tf_img = model.upscaled_img(scale)

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

def sr():

  saved_dir, mat_dir, bicubic_dir = val_img_path(opt.image, opt.scale, opt.output_dir)

  eng = matlab.engine.start_matlab()
  eng.addpath(r'./src/evaluation_mat', nargout=0)
  eng.addpath(r'./src/evaluation_mat/ifc-drrn')
  eng.addpath(r'./src/evaluation_mat/matlabPyrTools')

  aa = eng.get_ycbcr_image(opt.image, mat_dir, opt.scale);

  image_hash = sio.loadmat(mat_dir)
  img_y = image_hash['img_y']
  sr_img_y, elapsed_time = generator(img_y, opt.batch_size, opt.scale, channel=opt.channel, filter_num=opt.filter_num, model_name=opt.sr_method, model_path=opt.model_path, gpu_id=opt.gpu_id)

  image_hash['sr_img_y'] = sr_img_y
  sio.savemat(mat_dir, image_hash)

  eng.save_ycbcr_image(mat_dir, saved_dir, bicubic_dir);
  print("save image at {}, for {}s".format(saved_dir, elapsed_time))

  eng.quit()


if __name__ == '__main__':
  sr()
