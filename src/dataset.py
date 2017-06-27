import numpy as np
import h5py
from scipy.misc import imresize
import os

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

        self.gt_img = vars(self)["label_x{}".format(self.upscale)]
        self.total_size, self.channel, self.gt_height, self.gt_width = np.shape(self.gt_img)
        self.batch_ids = self.total_size // self.batch_size
        self.input_size = [self.gt_height//4, self.gt_width//4]

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
