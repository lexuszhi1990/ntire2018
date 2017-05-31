import cv2
import time, math
import numpy as np

from src.evaluation import psnr as compute_psnr
from src.evaluation import _SSIMForMultiScale as compute_ssim

gt_img_path = './dataset/test/set14/GT/face.png'
compared_img_path = './dataset/test/set14/lapsrn/v3/face_l4_lapsrn_x4.png'
bicubic_img_path = './dataset/test/set14/bicubic/face_l4_bicubic_x4.png'

gt_img = cv2.imread(gt_img_path)
upscaled_img = cv2.imread(compared_img_path)
bicubic_img = cv2.imread(bicubic_img_path)
# bicubic_img = cv2.resize(cv2.resize(gt_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

gt_img_expanded = np.expand_dims(gt_img, axis=0)
bicubic_img_expanded = np.expand_dims(bicubic_img, axis=0)
upscaled_img_expanded = np.expand_dims(upscaled_img, axis=0)

gt_img_yc = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCR_CB)
upscaled_img_yc = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2YCR_CB)
bicubic_img_yc = cv2.cvtColor(bicubic_img, cv2.COLOR_BGR2YCR_CB)

gt_img_yc_expanded = np.expand_dims(gt_img_yc, axis=0)
upscaled_img_yc_expanded = np.expand_dims(upscaled_img_yc, axis=0)
bicubic_img_yc_expanded = np.expand_dims(bicubic_img_yc, axis=0)

start_time = time.time()
psnr_predicted = compute_psnr(gt_img, upscaled_img)
psnr_bicubic = compute_psnr(gt_img, bicubic_img)
elapsed_time = time.time() - start_time
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

start_time = time.time()
yc_psnr_predicted = compute_psnr(gt_img_yc[:, :, 0], upscaled_img_yc[:, :, 0])
yc_psnr_bicubic = compute_psnr(gt_img_yc[:, :, 0], bicubic_img_yc[:, :, 0])
elapsed_time = time.time() - start_time
print("yc_PSNR_predicted=", yc_psnr_predicted)
print("yc_PSNR_bicubic=", yc_psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

start_time = time.time()
ssim_predicted = compute_ssim(gt_img_expanded, upscaled_img_expanded)
ssim_bicubic = compute_ssim(gt_img_expanded, bicubic_img_expanded)
elapsed_time = time.time() - start_time
print("ssim_predicted=", ssim_predicted)
print("ssim_bicubic=", ssim_bicubic)
print("It takes {}s for processing".format(elapsed_time))

start_time = time.time()
yc_ssim_predicted = compute_ssim(gt_img_yc_expanded, upscaled_img_yc_expanded)
yc_ssim_bicubic = compute_ssim(gt_img_yc_expanded, bicubic_img_yc_expanded)
elapsed_time = time.time() - start_time
print("yc ssim_predicted=", yc_ssim_predicted[0])
print("yc ssim_bicubic=", yc_ssim_bicubic[0])
print("It takes {}s for processing".format(elapsed_time))
