'''
usage:
  from src.dataset_builder.generate_tfrecords import create_tfrecord, generate_test_dataset

  create_tfrecord('/Users/david/mnt/data/set14', epoches=1, saved_name='py_train_t291_g100')

  generate_test_dataset('./dataset/py_test/bsd100/raw', './dataset/py_test/bsd100')

'''

import os
import numpy as np
from glob import glob
import tensorflow as tf

from src.pil_utils import *

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


def create_tfrecord(image_dir, epoches=1, saved_name='py_train', default_size=[256, 256], dst_dir='./dataset'):

    label_x8_list, label_x4_list, label_x2_list, data_list = [], [], [], []
    saved_path = os.path.join(dst_dir, "{}_x{}.tfrecords".format(saved_name, epoches))
    writer = tf.python_io.TFRecordWriter(saved_path)
    print 'generate_images at {}, saved dataset at {}'.format(image_dir, saved_path)

    for epoch in range(epoches):
      image_list = glob(os.path.join(image_dir, '*.*'))
      np.random.shuffle(image_list)
      for image_ab_path in image_list:
        image = augment(image_ab_path, default_size)

        resized_img = np.expand_dims(image, axis=0)
        label_x8 = resized_img

        resized_img = img_resize_float(image, 0.5)
        label_x4 = np.expand_dims(resized_img, axis=0)

        resized_img = img_resize_float(image, 0.25)
        label_x2 = np.expand_dims(resized_img, axis=0)

        resized_img = img_resize_float(image, 0.125)
        data = np.expand_dims(resized_img, axis=0)

        print 'epoch {}, image: {}'.format(epoch, image_ab_path)

        example = tf.train.Example(features=tf.train.Features(feature={
            'label_x8': _bytes_feature(label_x8.tobytes()),
            'label_x4': _bytes_feature(label_x4.tobytes()),
            'label_x2': _bytes_feature(label_x2.tobytes()),
            'data': _bytes_feature(data.tobytes())
        }))
        writer.write(example.SerializeToString())

    writer.close()

def read_tfrecord(filename, default_size=[256, 256]):
    filename ='./dataset/py_train_t291_g100_x1.tfrecords'
    filename_queue = tf.train.string_input_producer([filename, ])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label_x8': tf.FixedLenFeature([], tf.string),
          'label_x4': tf.FixedLenFeature([], tf.string),
          'label_x2': tf.FixedLenFeature([], tf.string),
          'data': tf.FixedLenFeature([], tf.string)
      }
    )
    label_x8 = tf.decode_raw(features['label_x8'], tf.float32)
    label_x8 = tf.reshape(label_x8, [default_size[0], default_size[1]])

    label_x4 = tf.decode_raw(features['label_x4'], tf.float32)
    label_x4 = tf.reshape(label_x4, [default_size[0]//2, default_size[1]//2])

    label_x2 = tf.decode_raw(features['label_x2'], tf.float32)
    label_x2 = tf.reshape(label_x2, [default_size[0]//4, default_size[1]//4])

    data = tf.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, [default_size[0]//8, default_size[1]//8])

z    # https://www.tensorflow.org/programmers_guide/reading_data#multiple_input_pipelines
    #  min_after_dequeue + (num_threads + a small safety margin) * batch_size
    # https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 40 * batch_size
    batch_label_x8, batch_label_x4, batch_label_x2, batch_data = tf.train.shuffle_batch([label_x8, label_x4, label_x2, data], batch_size = batch_size, num_threads = 4, min_after_dequeue = min_after_dequeue, capacity = capacity)

    return batch_label_x8, batch_label_x4, batch_label_x2, batch_data

if __name__ == '__main__':z
    sess = tf.InteractiveSession()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    label_x8_, label_x4_, label_x2_, data_ =sess.run([label_x8, label_x4, label_x2, data])

    fig = plt.figure()
    ax = plt.subplot("141")
    ax.imshow(label_x8_, cmap='gray')

    ax = plt.subplot("142")
    ax.imshow(label_x4_, cmap='gray')

    ax = plt.subplot("143")
    ax.imshow(label_x2_, cmap='gray')

    ax = plt.subplot("144")
    ax.imshow(data_, cmap='gray')

    plt.show()


