import os
from glob import glob
import cv2
import scipy.misc
import numpy as np

def im_resize(img, ratio, interp='bilinear'):
  return scipy.misc.cv2.resize(gt_img, None,fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)(img, ratio, interp=interp)

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

def shave(image, border=[0, 0]):
  height, width, _ = np.shape(image)
  return image[border[0]:height-border[0], border[1]:width-border[1], :]

def modcrop(image, modulo):
  height, width, _ = np.shape(image)
  height_modulo, width_modulo = height % modulo, width % modulo

  return image[0:height-height_modulo, 0:width-width_modulo, :]

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

def generate_test_dataset(dataset_path):
  gt_img_dir = os.path.join(dataset_path, 'GT')
  zoom_in_dir = os.path.join(dataset_path, 'lr_x2348')
  bicubic_dir = os.path.join(dataset_path, 'bicubic')

  if os.path.exists(zoom_in_dir) == False:
    os.system('rm -rf ' + zoom_in_dir)
  else:
    os.system('rm -rf ' + zoom_in_dir)
    os.mkdir(zoom_in_dir)

  if os.path.exists(bicubic_dir) == False:
    os.system('rm -rf ' + bicubic_dir)
  else:
    os.system('rm -rf ' + bicubic_dir)
    os.mkdir(bicubic_dir)

  scale_list = [2, 3, 4, 8];

  gt_img_list = glob(os.path.join(gt_img_dir, '*.*'))

  for image_ab_path in gt_img_list:
    image_basename = os.path.basename(image_ab_path).split('.')[0]
    dir_name = os.path.dirname(image_ab_path)

    image_raw = cv2.imread(image_ab_path)
    image_raw = modcrop(image_raw, 24)

    # update defalut image
    cv2.imwrite(image_ab_path, image_raw)
    print("update image {}".format(image_basename))

    for scale in scale_list:
      lr_img = cv2.resize(image_raw, None,fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_CUBIC)
      patch_name = '{}_l{}.png'.format(image_basename, scale)
      zoom_in_img_path = os.path.join(zoom_in_dir, patch_name)
      cv2.imwrite(zoom_in_img_path, lr_img)
      print("save lr image {} with scale {}".format(zoom_in_img_path, scale))

      bicubic_sr_img = cv2.resize(lr_img, None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
      patch_name = '{}_l{}_bicubic_x{}.png'.format(image_basename, scale, scale)
      bicubic_sr_img_path = os.path.join(bicubic_dir, patch_name)
      cv2.imwrite(bicubic_sr_img_path, bicubic_sr_img)
      print("save lr image {} with scale {}".format(bicubic_sr_img_path, scale))

# usage:
# for 'generate_test_dataset':
# from src.dataset_builder.generate_dataset import generate_test_dataset
# generate_test_dataset('./dataset/test/set14')
