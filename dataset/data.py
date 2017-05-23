import os
from glob import glob
import cv2
import scipy.misc
import numpy as np

def im_resize(img, ratio, interp='bilinear'):
  return scipy.misc.imresize(img, ratio, interp=interp)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float32)
  else:
    return scipy.misc.imread(path).astype(np.float32)

def center_crop(x, crop_h, crop_w=None):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return x[j:j+crop_h, i:i+crop_w]

# generate the training images
# http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees/16278334
def rotate_image(im, angle):
    if angle % 90 == 0:
        angle = angle % 360
        if angle == 0:
            return im
        elif angle == 90:
            return im.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            return im[::-1,::-1,:]
        elif angle == 270:
            return im.transpose((1,0, 2))[::-1,:,:]
    else:
        raise Exception('Error')

def transform(image):
  return np.array(image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def get_image(image_path, resize_height=256, resize_width=256, is_crop=False, is_grayscale=False):
  img = imread(image_path, is_grayscale=is_grayscale)

  return img

class DataSet(object):
  def __init__(self, path="./dataset/train", batch_size=16, image_size=[64,64], upscale_ratio=4):
    self.path = path
    self.batch_size = batch_size
    self.upscale_ratio = upscale_ratio
    self.image_size = image_size
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

  def finished(self):
    return self.idx == self.batch_idxs

  def restore(self):
    self.idx = 0
    np.random.shuffle(self.gt_imgs)

  def next(self):
    if self.idx == self.batch_idxs:
      return None

    batch_files = self.gt_imgs[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
    batch_gt = [get_image(batch_file, is_crop=self.is_crop) for batch_file in batch_files]
    batch_inputs = [im_resize(img, 1.0/self.upscale_ratio, interp='bicubic') for img in batch_gt]

    self.idx += 1

    return transform(batch_gt), transform(batch_inputs)

def generate_images(image_dir, base_dir, default_size=[312, 480]):
  print 'generate_images at ' + image_dir
  image_list = glob(image_dir + '*.*')
  for image_ab_path in image_list:
    image = cv2.imread(image_ab_path)
    origin_height, origin_width = image.shape[0], image.shape[1]
    if origin_height > origin_width:
      image = rotate_image(image, 90)

    # TODO: resize the image to default size
    crop_img = center_crop(image, default_size[0], default_size[1])

    img_d4 = cv2.resize(crop_img, None, fx=0.25, fy=0.25)
    image_basename = os.path.basename(image_ab_path).split('.')[0]
    img_gt_name = image_basename + '_gt.png'
    img_d4_name = image_basename + '_d4.png'
    cv2.imwrite(os.path.join(base_dir, img_gt_name), crop_img)
    # cv2.imwrite(os.path.join(base_dir, img_d4_name), img_d4)

    print(os.path.join(base_dir, img_gt_name))

def generate_train_images(rebuild=False):
  '''
    usage:
    from src.data import generate_train_images
  '''
  train_dir = './dataset/train'

  train_image_dirs = [
                      './dataset/coco_selected/',
                      './dataset/bsd300_train/'
                     ]

  if rebuild == True:
    os.system('rm -rf ' + train_dir)
  if os.path.exists(train_dir) == False:
    os.mkdir(train_dir)

  for image_dir in train_image_dirs:
    generate_images(image_dir, train_dir)

