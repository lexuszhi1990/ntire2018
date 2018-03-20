#!/usr/bin/python
'''
'''

import matlab
import matlab.engine
from src.utils import sess_configure, trainsform, transform_reverse
from src.eval_dataset import eval_dataset
from src.evaluation import shave_bd, compute_psnr, compute_ssim, compute_msssim

eng = matlab.engine.start_matlab()
eng.addpath(r'./src/evaluation_mat', nargout=0)
eng.addpath(r'./src/evaluation_mat/ifc-drrn')
eng.addpath(r'./src/evaluation_mat/matlabPyrTools')

def stop_matlab():
  eng.quit()

def load_img_from_mat(img_mat_path, scale):
  image_hash = sio.loadmat(img_mat_path)

  im_l_y = image_hash['label_x{}_y'.format(8//scale)]
  im_bicubic_ycbcr = image_hash['bicubic_l{}_x{}_ycbcr'.format(scale, scale)]
  im_bicubic_ycbcr = np.clip(im_bicubic_ycbcr*255., 0, 255.)
  img_gt = image_hash['label_x8_y']

  return im_l_y, im_bicubic_ycbcr, img_gt

def val_img_path(img_path, scale, sr_method, output_dir=None, verbose=False):
  img_name = os.path.basename(img_path).split('.')[0]
  upscaled_img_name =  "{}_l{}_{}_x{}.png".format(img_name, scale, sr_method, str(scale))
  upscaled_mat_name =  "{}_sr_x{}.mat".format(img_name, scale)
  bicubic_img_name =  "{}_bicubic_x{}.png".format(img_name, scale)

  if output_dir is not None and os.path.isdir(output_dir):
    base_dir = output_dir
  else:
    base_dir = os.path.dirname(img_path)

  if verbose is False:
    return os.path.join(base_dir, upscaled_img_name)
  else:
    return os.path.join(base_dir, upscaled_img_name), os.path.join(base_dir, upscaled_mat_name), os.path.join(base_dir, bicubic_img_name)

def save_img(image, img_path, scale, sr_method, output_dir):
  output_img_path = val_img_path(img_path)
  imsave(output_img_path, image)

  print("upscaled image size {}".format(np.shape(image)))
  print("save image at {}".format(output_img_path))

def save_mat(img, path, sr_method='edsr', scale=4):
  image_hash = sio.loadmat(path)
  # img_key = '{}_l{}_x{}_y'.format(sr_method, scale, scale)
  img_key = 'sr_img_y'
  image_hash[img_key] = img
  sio.savemat(path, image_hash)
  print('save mat at {} in {}'.format(path, img_key))

def matlab_validation(dataset_dir, sr_method, scale):
  dataset_dir_list = dataset_dir.split('/')[0:-1]
  base_dataset_dir = '/'.join(dataset_dir_list)

  eng = matlab.engine.start_matlab()

  eng.addpath(r'./src/evaluation_mat', nargout=0)
  eng.addpath(r'./src/evaluation_mat/ifc-drrn')
  eng.addpath(r'./src/evaluation_mat/matlabPyrTools')

  eng.eval_dataset_mat(base_dataset_dir, 'lapsrn/mat', sr_method, scale)
  eng.eval_dataset(base_dataset_dir, sr_method, scale)

  eng.quit()

def load_models(sr_method, model_path):
  # os.system('scp youlei@219.223.251.241:/home/youlei/workplace/srn_bishe/ckpt/EDSR_v215/EDSR_v215-epoch-1-step-19548-2017-10-12-13-44.ckpt-19548.index ./')

  # g_dir = './ckpt/' + sr_method
  g_dir = '/'.join(model_path.split('/')[:-1])
  print("load model from {}".format(g_dir))
  if tf.gfile.Exists(g_dir):
    # tf.gfile.DeleteRecursively(g_dir)
    return
  else:
    tf.gfile.MakeDirs(g_dir)

    # command = os.path.join('scp youlei@219.223.251.241:/home/youlei/workplace/srn_face/', model_path)
    command = os.path.join('scp youlei@219.223.251.241:/home/youlei/workplace/srn_bishe', model_path)
    os.system(command + '.index ' + g_dir)
    os.system(command + '.meta ' + g_dir)
    os.system(command + '.data-00000-of-00001 ' + g_dir)
    print(command)

def generator(input_img, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id):

  graph = tf.Graph()
  sess_conf = sess_configure(memory_per=.95)

  img_size = input_img.shape
  height, width = input_img.shape
  batch_images = np.zeros((batch_size, height, width, channel))
  batch_images[0, :, :, 0] = input_img

  with graph.as_default(), tf.Session(config=sess_conf) as sess:
    with tf.device("/gpu:{}".format(str(gpu_id))):

      inputs = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x2 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x4 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      gt_img_x8 = tf.placeholder(tf.float32, [batch_size, None, None, channel])
      is_training = tf.placeholder(tf.bool, [])

      SRNet = globals()[model_name]
      model = SRNet(inputs, gt_img_x2, gt_img_x4, gt_img_x8, image_size=img_size, upscale_factor=scale, filter_num=filter_num, is_training=is_training)
      model.init_gt_imgs()
      model.extract_features()
      model.reconstruct()
      upscaled_tf_img = model.get_image(scale)

      saver = tf.train.Saver()
      if os.path.isdir(model_path):
        latest_ckpt = tf.train.latest_checkpoint(model_path)
        if latest_ckpt:
          saver.restore(sess, latest_ckpt)
          print("restore model from dir %s"%latest_ckpt)
        else:
          print("cannot restore model from %s, plz checkout"%latest_ckpt)
      else:
        saver.restore(sess, model_path)
        print("restore model from file %s"%model_path)

      start_time = time.time()
      upscaled_img = sess.run(upscaled_tf_img, feed_dict={inputs: batch_images, is_training: False})
      elapsed_time = time.time() - start_time

      return upscaled_img[0], elapsed_time

def cal_ssim(upscaled_img_y, gt_img_y):
  gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
  upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
  ssim = compute_ssim(gt_img_ep, upscaled_img_ep)

  return ssim

def cal_msssim(upscaled_img_y, gt_img_y):
  gt_img_ep = np.expand_dims(np.expand_dims(gt_img_y, axis=0), axis=3)
  upscaled_img_ep = np.expand_dims(np.expand_dims(upscaled_img_y, axis=0), axis=3)
  msssim = compute_msssim(gt_img_ep, upscaled_img_ep)

  return msssim

def cal_image_index(gt_img_y, upscaled_img_y, scale):

  upscaled_img_y = np.clip(upscaled_img_y*255., 0, 255.)
  gt_img_y = np.clip(gt_img_y*255., 0, 255.)

  upscaled_img_y = shave_bd(upscaled_img_y, scale)
  gt_img_y = shave_bd(gt_img_y, scale)

  psnr = compute_psnr(upscaled_img_y, gt_img_y)
  ssim = cal_ssim(upscaled_img_y, gt_img_y)
  msssim = cal_msssim(upscaled_img_y, gt_img_y)

  return psnr, ssim, msssim

input_img = img_y
batch_size = 1
scale = 4
channel = 1
filter_num = 64
gpu_id = 3
pad = 2
sr_method= 'edsr'
model_name = 'LapSRN_v7'
model_path ='./saved_models/x4/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774'
output_dir = '.'



img_path = '0801x4d.png'


saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, sr_method, output_dir=None, verbose=True)
eng.get_ycbcr_image(img_path, mat_dir, scale);
image_hash = sio.loadmat(mat_dir)
input_img = image_hash['img_y']

height, width = input_img.shape
hr_img = np.zeros((height*4, width*4))

img_1 = input_img[:height/3+pad,:width/3+pad]
upscaled_img = generator(img_1, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_1 = upscaled_img[0][:upscaled_img[0].shape[0]-pad*4, :upscaled_img[0].shape[1]-pad*4, 0]
hr_img[:height/3*4,:width/3*4] = upscaed_img_1

img_2 = input_img[:height/3+2,width/3-1:width/3*2+1]
upscaled_img = generator(img_2, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_2 = upscaled_img[0][:upscaled_height-pad*4, 4:upscaled_width-4, 0]
hr_img[:height/3*4,width/3*4:width/3*2*4] = upscaed_img_2

img_3 = input_img[:height/3,width/3*2:]
upscaled_img = generator(img_3, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_3 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[:height/3*4,width/3*2*4:] = upscaed_img_3

img_4 = input_img[height/3:height/3*2,:width/3]
upscaled_img = generator(img_4, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_4 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,:width/3*4] = upscaed_img_4

img_5 = input_img[height/3:height/3*2,width/3:width/3*2]
upscaled_img = generator(img_5, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_5 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*4:width/3*2*4] = upscaed_img_5

img_6 = input_img[height/3:height/3*2,width/3*2:]
upscaled_img = generator(img_6, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_6 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*4:height/3*2*4,width/3*2*4:] = upscaed_img_6

img_7 = input_img[height/3*2:,:width/3]
upscaled_img = generator(img_7, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_7 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,:width/3*4] = upscaed_img_7

img_8 = input_img[height/3*2:,width/3:width/3*2]
upscaled_img = generator(img_8, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_8 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*4:width/3*2*4] = upscaed_img_8

img_9 = input_img[height/3*2:,width/3*2:]
upscaled_img = generator(img_9, batch_size, scale, channel, filter_num, model_name, model_path, gpu_id)
upscaled_height, upscaled_width, _ = upscaled_img[0].shape
upscaed_img_9 = upscaled_img[0][:upscaled_height, :upscaled_width, 0]
hr_img[height/3*2*4:,width/3*2*4:] = upscaed_img_9


def SR(dataset_dir, batch_size, scale, channel, filter_num, sr_method, model_path, gpu_id):

  dataset_image_path = os.path.join(dataset_dir, '*.mat')

  PSNR = []
  SSIM = []
  MSSSIM = []
  EXEC_TIME = []

  # scale_list = [2, 4, 8]
  # for scale in scale_list[0:np.log2(init_scale).astype(int)]:
  psnrs = []
  ssims = []
  msssims = []
  exec_time = []

  for filepath in glob(dataset_image_path):

    tf.reset_default_graph()

    im_l_y, im_h_ycbcr, img_gt_y = load_img_from_mat(filepath, scale)
    im_h_y, elapsed_time = generator(im_l_y, batch_size, scale, channel, filter_num, sr_method, model_path, gpu_id)
    save_mat(im_h_y, filepath, sr_method, scale)

    psnr, ssim, msssim = cal_image_index(img_gt_y, im_h_y[:,:,0], scale)
    psnrs.append(psnr)
    ssims.append(ssim)
    msssims.append(msssim)
    exec_time.append(elapsed_time)

    print("for image %s, scale: %d, average exec time: %.4fs\n-- PSNR/SSIM/MSSSIM: %.4f/%.4f/%.4f\n"%(filepath, scale, elapsed_time, psnr, ssim, msssim))

  PSNR.append(psnrs)
  SSIM.append(ssims)
  MSSSIM.append(msssims)
  EXEC_TIME.append(exec_time)

  return PSNR, SSIM, MSSSIM, EXEC_TIME



img_path = '0801x4d.png'
scale = 4
output_dir = '.'

saved_dir, mat_dir, bicubic_dir = val_img_path(img_path, scale, output_dir)

null = eng.get_ycbcr_image(img_path, mat_dir, scale);
image_hash = sio.loadmat(mat_dir)
img_y = image_hash['img_y']
save_mat(hr_img, mat_dir)
null = eng.save_ycbcr_image(mat_dir, saved_dir, bicubic_dir)
