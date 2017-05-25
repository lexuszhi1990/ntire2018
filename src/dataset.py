import numpy as np
import h5py
import cv2

class DatasetFromHdf5(object):
  def __init__(self, path="./dataset/train.h5", batch_size=10, upscale=4):
    self.path = path
    self.batch_size = batch_size
    self.upscale = upscale
    self.hf = h5py.File(self.path)
    self.data = self.hf.get('data')
    self.label = self.hf.get('label')

  def transform(self, image):
    return np.array(image)/127.5 - 1.

  def batch_transpose(self,images):
    return [image.T for image in images]

  def batch_resize(self, images):
    aa = [cv2.resize(image, None, fx=1.0/self.upscale, fy=1.0/self.upscale,interpolation=cv2.INTER_CUBIC) for image in images]
    if len(np.shape(aa)) == 3:
      aa = [np.expand_dims(x, axis=-1) for x in aa]
    return aa

  def next(self, index):
    batch_label = self.label[index*self.batch_size:(index+1)*self.batch_size]
    batch_label = self.batch_transpose(batch_label)
    batch_data = self.batch_resize(batch_label)

    return self.transform(batch_data), self.transform(batch_label)
