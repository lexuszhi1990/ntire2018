import numpy as np
import h5py
from scipy.misc import imresize
import os

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

class TrainDatasetFromHdf5(object):
    def __init__(self, file_path, batch_size=8, upscale=4):
        self.batch_size = batch_size
        self.upscale = upscale
        self.file_path = file_path
        self.current_id = 0

        self.init()

    def init(self):
        hf = h5py.File(self.file_path)
        if hf.get("label_x8") == None:
            self.label_x2 = hf.get("label_x4")
            self.label_x2 = hf.get("label_x2")
            self.data = hf.get("data")
        else:
            self.label_x8 = hf.get("label_x8")
            self.label_x4 = hf.get("label_x4")
            self.label_x2 = hf.get("label_x2")
            self.data = hf.get("data")

        self.input_size = [self.data.shape[2], self.data.shape[3]]

        self.gt_img = vars(self)["label_x{}".format(self.upscale)]
        self.total_size, self.channel, self.gt_height, self.gt_width = np.shape(self.gt_img)
        self.batch_ids = self.total_size // self.batch_size

    def batch_transpose(self,images):
        return np.array([image.T for image in images])

    def transform(self, image):
        return np.array(image)/255.0

    def next_batch(self, current_index=-1):

        if current_index == -1:
            if self.is_finished():
                index = np.random.randint(self.batch_ids-1)
            else:
                index = self.current_id
        else:
            index = current_index


        batch_label_x8 = self.batch_transpose(self.label_x8[index*self.batch_size:(index+1)*self.batch_size])
        batch_label_x4 = self.batch_transpose(self.label_x4[index*self.batch_size:(index+1)*self.batch_size])
        batch_label_x2 = self.batch_transpose(self.label_x2[index*self.batch_size:(index+1)*self.batch_size])
        batch_data = self.batch_transpose(self.data[index*self.batch_size:(index+1)*self.batch_size])

        self.current_id += 1

        return [batch_label_x8, batch_label_x4, batch_label_x2, batch_data]

    def is_finished(self):
        return self.current_id >= self.batch_ids

    def reset_anchor(self):
        self.current_id = 0

    def rebuild(self):
        print('rebuild the dataset...')
        os.system('matlab -nodesktop -nosplash -r train_h5_eval 1>/dev/null');
