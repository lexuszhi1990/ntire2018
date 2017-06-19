'''
usage:
  from src.dataset_builder.select_img_from_coco import select_img_from_coco
'''
import sys
import os
import cv2

coco_path = '/home/mcc207/datasets/coco/'
sys.path.append(os.path.join(coco_path, 'PythonAPI'))
from pycocotools.coco import COCO

def select_img_from_coco(coco_selected_img_path = '/home/mcc207/datasets/coco_selected', cat_num=40):
  if os.path.exists(coco_selected_img_path) == False:
    os.mkdir(coco_selected_img_path)
  else:
    os.system('rm -rf ' + coco_selected_img_path)
    os.mkdir(coco_selected_img_path)

  annFile = os.path.join(coco_path, 'annotations', 'instances_train2014.json')

  coco=COCO(annFile)
  cats = coco.loadCats(coco.getCatIds())
  cat_names=[cat['name'] for cat in cats]
  cat_ids = coco.getCatIds(cat_names)

  for cat_id in cat_ids:
    img_ids = coco.getImgIds(catIds=cat_id );
    print(cat_id)
    save_imgs = 0
    for image_id in img_ids:
      img = coco.loadImgs(image_id)[0]

      image = cv2.imread(os.path.join(coco_path, 'images/train2014', img['file_name']))

      if (img['width'] > 420 and img['height'] > 312) or (img['height'] > 420 and img['width'] > 312):
        dst_image_path = os.path.join(coco_selected_img_path, img['file_name'].split('.')[0] + '.png')
        cv2.imwrite(dst_image_path, image)
        print(dst_image_path)

        save_imgs += 1

      if save_imgs > cat_num:
        break
