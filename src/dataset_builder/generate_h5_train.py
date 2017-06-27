'''
usage:
  from src.dataset_builder.generate_h5_train import generate_train_h5
  generate_train_h5('/Users/david/mnt/data/set14/', epoches=4, saved_name='py_train')
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
