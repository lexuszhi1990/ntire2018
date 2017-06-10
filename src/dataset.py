import numpy as np
import h5py
from scipy.misc import imresize

class TrainDataset(object):
  def __init__(self, file_path, train_img_size=[128,128], batch_size=10, upscale=4):
    self.file_path = file_path
    self.batch_size = batch_size
    self.upscale = upscale
    self.gt_height, self.gt_width = train_img_size

    self.hf = h5py.File(self.file_path)
    self.data = self.hf.get('data')
    self.label = self.hf.get('label')
    self.len = self.data.len()

    _, self.channel, self.max_height, self.max_width = np.shape(self.data)
    self.batch_ids = self.data.len() // self.batch_size
    self.input_image_size = [self.gt_height//self.upscale, self.gt_width//self.upscale]

  def transform(self, image):
    return np.array(image)/255.0

  def batch_transpose(self,images):
    return np.array([image.T for image in images])

  def batch_resize(self, images):
    aa = [imresize(image, 1.0/self.upscale, interp='bilinear') for image in images]
    if len(np.shape(aa)) == 3:
      aa = [np.expand_dims(x, axis=-1) for x in aa]
    return aa

  def finished(self, step):
    return step >= self.data.len()/self.batch_size

  def next(self, index):
    batch_label = self.label[index*self.batch_size:(index+1)*self.batch_size]
    batch_label = self.batch_transpose(batch_label)
    batch_input = self.batch_resize(batch_label)

    # return batch_input, batch_label
    return self.transform(batch_input), self.transform(batch_label)

class DatasetFromHdf5(object):
    def __init__(self, file_path, batch_size=8, upscale=4):
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

        self.batch_size = batch_size
        self.upscale = upscale
        self.len = self.data.len()
        self.batch_ids = self.data.len() // self.batch_size

        self.input_image_size = [self.data.shape[2], self.data.shape[3]]
        _, self.channel, self.gt_height, self.gt_width = np.shape(self.label_x4)

    def batch_transpose(self,images):
        return np.array([image.T for image in images])

    def next(self, index):
        batch_data = self.data[index*self.batch_size:(index+1)*self.batch_size]
        batch_label_x2 = self.label_x2[index*self.batch_size:(index+1)*self.batch_size]
        batch_label_x4 = self.label_x4[index*self.batch_size:(index+1)*self.batch_size]

        return self.batch_transpose(batch_data), self.batch_transpose(batch_label_x2), self.batch_transpose(batch_label_x4)

    def __len__(self):
        return self.data.shape[0]
