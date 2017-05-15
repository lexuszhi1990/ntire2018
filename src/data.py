import os
from glob import glob
import cv2
import scipy.misc
import numpy as np

def im_resize(img, ratio):
  return scipy.misc.imresize(img, ratio)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float32)
  else:
    return scipy.misc.imread(path).astype(np.float32)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image):
  return np.array(image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def get_image(image_path, resize_height=256, resize_width=256, is_crop=False, is_grayscale=False):
  img = imread(image_path, is_grayscale=is_grayscale)
  if is_crop:
    img = center_crop(img, img.shape[0], img.shape[1], resize_height, resize_width)

  return img

class DataSet(object):
  def __init__(self, path="./dataset/train", batch_size=16, upscale_ratio=4):
    self.path = path
    self.upscale_ratio = upscale_ratio
    self.batch_size = batch_size
    self.is_crop = False
    self.shuffle = False
    self.is_grayscale = False

    self.gt_imgs = None
    self.batch_idxs = -1
    self.idx = 0

    self.read()

  def read(self):
    self.gt_imgs = glob(os.path.join(self.path, '*_gt.png'))
    self.batch_idxs = len(self.gt_imgs) // self.batch_size

  def next(self):
    if self.idx == self.batch_idxs:
      return None

    batch_files = self.gt_imgs[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
    batch_gt = [get_image(batch_file, is_crop=self.is_crop) for batch_file in batch_files]
    batch_inputs = [im_resize(img, 1.0/self.upscale_ratio) for img in batch_gt]

    self.idx += 1

    return transform(batch_gt), transform(batch_inputs)

# generate the training images
def generate_images(file_op, image_dir, base_dir='./', is_trainning=True):

  print 'generate_images at ' + image_dir
  image_list = glob(image_dir + '*.*')
  for image_ab_path in image_list:
    image = cv2.imread(image_ab_path)
    origin_height, origin_width = image.shape[0], image.shape[1]
    default_height, default_width = origin_height - origin_height%12, origin_width - origin_width%12
    crop_img = image[0:default_height, 0:default_width]

    img_d4 = cv2.resize(crop_img, None, fx=0.25, fy=0.25)
    image_basename = os.path.basename(image_ab_path).split('.')[0]
    img_gt_name = image_basename + '_gt.png'
    img_d4_name = image_basename + '_d4.png'
    cv2.imwrite(os.path.join(base_dir, img_gt_name), crop_img)
    cv2.imwrite(os.path.join(base_dir, img_d4_name), img_d4)
    print(img_gt_name)
    file_op.write(img_gt_name + '\t' + img_d4_name + '\n')

    print os.path.join(base_dir, img_gt_name)

def generate_train_images(rebuild=False):
  train_dir = './dataset/train'

  train_image_dirs = [
                      '/Users/david/mnt/dataset/image_SRF_4/'
                     ]

  if rebuild == True:
    os.system('rm -rf ' + train_dir)
  if os.path.exists(train_dir) == False:
    os.mkdir(train_dir)

  train_file_op = open('./dataset/train.txt','w')

  for image_dir in train_image_dirs:
    generate_images(train_file_op, image_dir, train_dir)

  train_file_op.close()
