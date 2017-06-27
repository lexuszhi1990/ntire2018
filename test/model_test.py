import tensorflow as tf
from tensorflow.python.ops import random_ops

from src.model import LapSRN_v1, LapSRN_v2

batch_size = 2
gt_shape = [32, 32]
upscale_factor=4
in_shape = [x/upscale_factor for x in gt_shape]
reg=0.0001
filter_num=64


tf.reset_default_graph()
inputs = random_ops.random_uniform([batch_size, in_shape[0], in_shape[1], 1], dtype=tf.float32, seed=1)
labels_x2 = random_ops.random_uniform([batch_size, in_shape[0]*2, in_shape[1]*2, 1], dtype=tf.float32, maxval=100)
labels_x4 = random_ops.random_uniform([batch_size, in_shape[0]*4, in_shape[1]*4, 1], dtype=tf.float32, maxval=100)
labels_x8 = random_ops.random_uniform([batch_size, in_shape[0]*8, in_shape[1]*8, 1], dtype=tf.float32, maxval=100)
batch_inputs = tf.placeholder(tf.float32, [batch_size, None, None, 1])
batch_gt_x2 = tf.placeholder(tf.float32, [batch_size, None, None, 1])
batch_gt_x4 = tf.placeholder(tf.float32, [batch_size, None, None, 1])
batch_gt_x8 = tf.placeholder(tf.float32, [batch_size, None, None, 1])
is_training = tf.placeholder(tf.bool, [])

model = LapSRN_v2(batch_inputs, batch_gt_x2, batch_gt_x4, batch_gt_x8, image_size=in_shape, is_training=is_training, upscale_factor=upscale_factor, reg=reg, filter_num=filter_num)
model.init_vars()
model.extract_features()
model.reconstruct()
loss = model.l1_loss()

sess = tf.InteractiveSession()
all_variables = tf.global_variables()
sess.run(tf.variables_initializer(set(all_variables)))

batch_in, batch_img_x2, batch_img_x4, batch_img_x8 = sess.run([inputs, labels_x2, labels_x4, labels_x8])
sess.run([loss], feed_dict={batch_gt_x2: batch_img_x2, batch_gt_x4: batch_img_x4, batch_gt_x8: batch_img_x8, batch_inputs: batch_in, is_training: True})
