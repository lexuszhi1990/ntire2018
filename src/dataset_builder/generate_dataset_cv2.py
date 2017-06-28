'''
usage:
  from src.dataset_builder.generate_dataset_new import generate_train_h5, generate_test_dataset

  generate_train_h5('/Users/david/mnt/data/set14/', epoches=4, saved_name='py_train')

  generate_test_dataset('../../../datasets/set14/', './dataset/test/set14')

'''

import os
import h5py
import numpy as np
from glob import glob

from src.cv2_utils import *

def random_interp_resize(image, scale):
    # scale_list = [0.5, 0.25, 0.125]
    # interp_list = ['nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic']
    # scipy_imresize(image, size, interp='bilinear')
    # img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)

    interp_list = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]
    interp = interp_list[np.random.randint(0, len(interp_list))]
    img = cv2_imresie(image, scale, scale, interp)

    return img

def augment(img, gt_size):
    image = img_read(img)
    image = normalize(image)
    image = random_resize(image)
    image = varify_size(image, gt_size)
    image = random_crop(image, gt_size)
    image = random_flip_left_right(image)
    image = extract_y_channel(image)

    return image

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

            resized_img = random_interp_resize(image, 0.5)
            label_x4_list.append(np.expand_dims(resized_img, axis=0))

            resized_img = random_interp_resize(image, 0.25)
            label_x2_list.append(np.expand_dims(resized_img, axis=0))

            resized_img = random_interp_resize(image, 0.125)
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
    image_basename = os.path.basename(image_ab_path).split('.')[0]

    image_raw = cv2.imread(image_ab_path)
    image_raw = modcrop(image_raw, 24)

    # update defalut image
    image_gt_path = os.path.join(gt_img_dir, "{}.png".format(image_basename))
    cv2.imwrite(image_gt_path, image_raw)
    print("update gt image {}".format(image_basename))

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
