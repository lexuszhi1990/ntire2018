import numpy as np
import h5py

class DatasetFromHdf5(object):
  def __init__(self, path="./dataset/train.h5", batch_size=10):
    self.path = path
    self.batch_size = batch_size
    self.hf = h5py.File(self.path)
    self.data = hf.get('data')
    self.label = hf.get('label')

    def transform(self, image):
      return np.array(image)/127.5 - 1.

    def next(self, index):
      batch_data = self.data[index*self.batch_size:(index+1)*self.batch_size]
      batch_label = self.label[index*self.batch_size:(index+1)*self.batch_size]

      return transform(batch_data), transform(batch_inputs)
