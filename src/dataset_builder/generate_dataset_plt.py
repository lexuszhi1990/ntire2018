'''
usage:
  from src.dataset_builder.generate_dataset_plt import generate_train_h5, generate_test_dataset

  generate_train_h5('/Users/david/mnt/data/set14/', epoches=4, saved_name='py_train')

  generate_test_dataset('../../../datasets/set14/', './dataset/test/set14')

'''

import os
import h5py
import numpy as np
from glob import glob

from src.cv2_utils import *

def generate_train_h5(image_dir, epoches=1, saved_name='py_train', default_size=[256, 256], dst_dir='./dataset'):

    label_x8_list, label_x4_list, label_x2_list, data_list = [], [], [], []
    saved_path = os.path.join(dst_dir, "{}_x{}.h5".format(saved_name, epoches))
    print 'generate_images at {}, saved h5 at {}'.format(image_dir, saved_path)

    for epoch in range(epoches):
        image_list = glob(os.path.join(image_dir, '*.*'))
        for image_ab_path in image_list:
            image = augment(image_ab_path, default_size)

            resized_img = np.expand_dims(image, axis=0)
            label_x8_list.append(resized_img)

            resized_img = img_resize_float(image, 0.5)
            label_x4_list.append(np.expand_dims(resized_img, axis=0))

            resized_img = img_resize_float(image, 0.25)
            label_x2_list.append(np.expand_dims(resized_img, axis=0))

            resized_img = img_resize_float(image, 0.125)
            data_list.append(np.expand_dims(resized_img, axis=0))


    f = h5py.File(saved_path, "w")
    f.create_dataset("label_x8", data=label_x8_list, chunks=True)
    f.create_dataset("label_x4", data=label_x4_list, chunks=True)
    f.create_dataset("label_x2", data=label_x2_list, chunks=True)
    f.create_dataset("data", data=data_list, chunks=True)
    f.close()

    return label_x8_list, label_x4_list, label_x2_list, data_list

def generate_test_dataset(gt_dir, dataset_path):
  gt_img_dir = os.path.join(dataset_path, 'GT')
  zoom_in_dir = os.path.join(dataset_path, 'lr_x2348')
  bicubic_dir = os.path.join(dataset_path, 'bicubic')

  if os.path.exists(gt_img_dir) == False:
    os.system('rm -rf ' + gt_img_dir)
  else:
    os.system('rm -rf ' + gt_img_dir)
    os.mkdir(gt_img_dir)

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

  gt_img_list = glob(os.path.join(gt_dir, '*.*'))

  for image_ab_path in gt_img_list:
    image_basename, ext = os.path.splitext(image_ab_path)

    image = img_read(image_ab_path)
    image = centered_mod_crop(image, 24)

    # update defalut image
    image_gt_path = os.path.join(gt_img_dir, "{}.png".format(image_basename))
    image.save(image_gt_path)
    print("update gt image {}".format(image_gt_path))

    for scale in scale_list:
      lr_img = img_resize_float(image, 1.0/scale)
      patch_name = '{}_l{}.png'.format(image_basename, scale)
      zoom_in_img_path = os.path.join(zoom_in_dir, patch_name)
      lr_img.save(zoom_in_img_path)
      print("save lr image {} with scale {}".format(zoom_in_img_path, scale))

      bicubic_sr_img = img_resize_float(lr_img, scale)
      patch_name = '{}_l{}_bicubic_x{}.png'.format(image_basename, scale, scale)
      bicubic_sr_img_path = os.path.join(bicubic_dir, patch_name)
      bicubic_sr_img.save(bicubic_sr_img_path)
      print("save lr image {} with scale {}".format(bicubic_sr_img_path, scale))
