import numpy as np
import cv2

def cv2_img_read(img_path):
    return cv2.imread(img_path)

def cv2_imresie(image, fx, fy, interpolation):
  # Interpolation to use for re-sizing [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]
  image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)

  return image

def varify_size(image, size):
    img_h, img_w = image.shape[:2]

    if img_h < size[0] or img_w < size[1]:
        fy, fx = size[0]/np.float(img_h)+0.1, size[1]/np.float(img_w)+0.1
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    return image

def random_flip_left_right(image):
    if np.random.randint(2) == 1:
        image = cv2.flip(image, 1)
    return image

def random_brightness(image, alpha=2.0):
    gamma = np.random.rand() * alpha
    gf = [[255 * pow(i/255, 1/gamma)] for i in range(256)]
    table = np.reshape(gf, (256, -1))
    image = cv2.LUT(image, table)
    return image

def random_crop_and_zoom(image, dst_size=[256, 256], alpha=0.6):
    img_h, img_w = image.shape[:2]
    r_scale = np.random.uniform(alpha, 1)
    scaled_size = [int(dst_size[0]*r_scale), int(dst_size[1]*r_scale)]

    h_rate = 1 - scaled_size[0]/np.float(img_h)
    w_rate = 1 - scaled_size[1]/np.float(img_w)
    v1 = np.floor(np.random.uniform(0, h_rate) * scaled_size[0])
    v2 = np.floor(np.random.uniform(0, w_rate) * scaled_size[1])

    image = image[v1:(v1+scaled_size[0]), v2:(v2+scaled_size[1]), :]
    image = cv2.resize(image, tuple(dst_size))
    return image

def random_crop(image, dst_size=[256, 256], alpha=0.6):
    img_h, img_w = image.shape[:2]
    h_rate = 1 - dst_size[0]/np.float(img_h)
    w_rate = 1 - dst_size[1]/np.float(img_w)
    v1 = np.floor(np.random.uniform(0, h_rate) * dst_size[0]).astype(int)
    v2 = np.floor(np.random.uniform(0, w_rate) * dst_size[1]).astype(int)
    image = image[v1:(v1+dst_size[0]), v2:(v2+dst_size[1]), :]

    return image

def random_zoom(image, alpha=0.1):
    img_h, img_w = image.shape[:2]
    r = np.random.uniform(0, alpha)
    v1 = np.random.randint(0, int(r*img_h)) if (int(r*img_h) != 0) else 0
    v2 = np.random.randint(0, int(r*img_w)) if (int(r*img_w) != 0) else 0
    image = image[v1:(v1+int((1-r)*img_h)), v2:(v2+int((1-r)*img_w)), :]
    image = cv2.resize(image, (img_h, img_w))
    return image

def random_resize(image, alpha=0.6):
  r = np.random.uniform(1, alpha)
  image = cv2.resize(image, None, fx=r, fy=r,interpolation=cv2.INTER_CUBIC)

  return image

def cvt_ycrcb(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

def extract_y_channel(image):
  return cvt_ycrcb(image)[:,:,0]

def center_crop(x, crop_h, crop_w=None):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return x[j:j+crop_h, i:i+crop_w]

def shave(image, border=[0, 0]):
  height, width, _ = np.shape(image)
  return image[border[0]:height-border[0], border[1]:width-border[1], :]

def modcrop(image, modulo):
  height, width, _ = np.shape(image)
  height_modulo, width_modulo = height % modulo, width % modulo

  return image[0:height-height_modulo, 0:width-width_modulo, :]

def normalize(image):
    image = image / 255.0
    return np.array(image).astype(np.float32)
