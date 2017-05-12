from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
from glob import glob
import scipy.misc
import numpy as np

def imread(path):
  return scipy.misc.imread(path).astype(np.float32)

def imsave(path, image):
  return scipy.misc.imsave(path, image)

def transform(image):
  return np.array(image)/127.5 - 1.

def inverse_transform(image):
  return (image+1.)/2.

def generate_image_path(ds_dir, dataset, img_base_name, upscale_factor, sr_method, img_type='.png'):
  return os.path.join(ds_dir, dataset, '%s_srf_%d_%s%s'%(img_base_name, upscale_factor, sr_method, img_type))

def dataset_images(ds_dir, dataset, data_type='*_gt', img_type='.png'):
  return glob(os.path.join(ds_dir, dataset, data_type + img_type))

def center_crop_image_scipy(x, image_size):
  croped_h, croped_w = image_size
  h, w = x.shape[:2]
  j = int(round((h - croped_h)/2.))
  i = int(round((w - croped_w)/2.))
  if(i > 0 and j > 0):
    x = x[j:j+croped_h, i:i+croped_w]
    return scipy.misc.imresize(x, image_size)
  else:
    return x

def load_image(image_path, image_size=None, centerd_crop=False, norm=True, expand_dims=False):
  image = imread(image_path)
  if centerd_crop:
    image = center_crop_image_scipy(image, image_size)
  if norm:
    image = transform(image)
  if expand_dims:
    image = np.expand_dims(image, 0)

  return image

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

def save_images(image_path, images, size=[1,1]):
  merged_image = merge(inverse_transform(images), size)
  return imsave(image_path, merged_image)

def load_dataset(ds_dir, dataset, data_type='*_gt', img_type='.png'):
  images = glob(os.path.join(ds_dir, dataset, data_type + img_type))
  image_list = []
  for image in images:
    img = load_image(image, None, resize=False)
    image_list.append(img)

  return np.array(image_list)
