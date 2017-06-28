'''
usage:
  import src.pil_utils as pil_utils
  from src.pil_utils import *

note:
  the size in PLT Image model are (width, height).
'''

import numpy as np
from PIL import Image

def img_read(image_path):
  # http://effbot.org/imagingbook/image.htm
  image = Image.open(image_path)

  return image

def img_resize(image, size, interp=Image.BICUBIC):
  # [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS]
  img_new = image.resize(size, interp)

  return img_new

def cvt_ycrcb(image):
  return image.convert('YCbCr')

def extract_y_channel(image):
  img_arr = np.array(cvt_ycrcb(image))
  image = Image.fromarray(img_arr[:,:,0])

  return image

def random_interp():
  interp_list = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS]
  interp = interp_list[np.random.randint(0, len(interp_list))]

  return interp

def normalize(image):
  image = np.array(image) / 255.0
  image = Image.fromarray(image)
  return image

def random_resize(image, alpha=0.6):
  r = np.random.uniform(1, alpha)
  width, height = image.size
  size = (int(r*width), int(r*height))
  image = img_resize(image, size, random_interp())

  return image

def img_resize_float(image, alpha, interp=Image.BICUBIC):
  width, height = image.size
  size = (int(alpha*width), int(alpha*height))
  img = img_resize(image, tuple(size), interp)

  return img

def varify_size(image, size):
  img_w, img_h = image.size

  if img_h < size[0] or img_w < size[1]:
    image = img_resize(image, tuple(size), Image.BILINEAR)

  return image

def random_crop(image, dst_size=[256, 256], alpha=0.6):
  '''
    Returns a copy of a rectangular region from the current image. The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
  '''
  img_w, img_h = image.size
  w_rate = 1 - dst_size[0]/np.float(img_w)
  h_rate = 1 - dst_size[1]/np.float(img_h)
  v1 = np.floor(np.random.uniform(0, w_rate) * dst_size[0]).astype(int)
  v2 = np.floor(np.random.uniform(0, h_rate) * dst_size[1]).astype(int)

  img_new = image.crop((v1, v2, v1+dst_size[0], v2+dst_size[1]))

  return img_new

def random_flip_left_right(image):
  image = np.array(image)

  if np.random.randint(2) == 1:
      image = np.fliplr(image)
  if np.random.randint(2) == 1:
      image = np.flipud(image)

  image = Image.fromarray(image)

  return image

def random_rotate(image):
  angle = np.random.randint(1, 5) * 90
  image = image.rotate(angle)

  return image

def centered_mod_crop(img, modulo):
  width, height = img.size   # Get dimensions
  new_width = width - width % modulo
  new_height = height - height % modulo
  left = (width - new_width)/2
  top = (height - new_height)/2
  right = (width + new_width)/2
  bottom = (height + new_height)/2

  return img.crop((left, top, right, bottom))

def augment(img_path, gt_size):
  image = img_read(img_path)
  image = extract_y_channel(image)
  image = random_resize(image)
  image = varify_size(image, gt_size)
  image = random_crop(image, gt_size)
  image = random_flip_left_right(image)
  image = normalize(image)

  return image

