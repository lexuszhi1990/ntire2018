# code for NTIRE 2018 image super-resolution Challenge

### dependencies

- main framework: tensorflow: 1.2 or above
- matlab python api

### how to install matlab python api

`matlabroot` is the matlab root path.

```
cd "matlabroot/extern/engines/python"
python setup.py install
```

then we load eval matlab commands in py env:
```
>>> import matlab
>>> matlab.__path__
['/usr/local/lib/python2.7/dist-packages/matlab']
```

references:
https://cn.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

### demo

CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 --batch_size=1 --channel=1 --filter_num=64 --scale=4 --sr_method=EDSR_v313 --model_path=./ckpt/EDSR_v313/EDSR_v313-epoch-1-step-20000-2018-02-12-02-56.ckpt-20000 --image=./demo/test.png --output_dir=./ --scale=4

### train

```
CUDA_VISIBLE_DEVICES=3 python solver.py --gpu_id=3 --dataset_dir=./dataset/div2k_difficulty_x50.h5 --g_log_dir=./log/LapSRN_v7_c --g_ckpt_dir=./ckpt/LapSRN_v7_c --default_sr_method='LapSRN_v7' --test_dataset_path=./dataset/mat_test/set5/mat --epoches=1 --inner_epoches=1 --default_channel=1 --upscale_factor=4 --filter_num=64 --continued_training --defalut_model_path=./ckpt/LapSRN_v7/LapSRN_v7-epoch-2-step-9774-2017-07-23-13-59.ckpt-9774 --batch_size=2
```

### datasets and models

the image dataset and models are uploaded on [baiduyun]()

### results

the (pnsr/ssim/ifc) results on set5/set14/bsd100 results:

|method | upscale_factor | set5 | set14 | bsd100|
|:-----:|:--------------:|:------|:-------|:----|
|SRCNN  | 4              |30.49/0.862/2.997|27.61/0.754/2.767|26.91/0.712/2.412|
|DRRN   | 4              |31.68/0.889/3.702|28.31/0.774/3.254|27.38/0.728/2.746|
|LapSRN | 4              |31.54/0.885/3.559|28.19/0.772/3.147|27.32/0.728/2.677|
|ours   |4               |31.67/0.890/3.620|28.26/0.774/3.207|27.36/0.729/2.698|
|SRCNN  | 8              |25.33/0.689/0.938|23.85/0.593/0.865|24.13/0.565/0.705|
|LapSRN | 8              |26.14/0.738/1.302|24.44/0.623/1.134|24.54/0.586/0.893|
|ours   | 8              |26.22/0.747/1.340|24.58/0.627/1.193|24.61/0.588/0.928|

### references

- Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014
- Shi W, Caballero J, Huszar F, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network[C]// CVPR  2016:1874-1883.
- Accurate image super-resolution using very deep convolutional networks[C] //Computer Vision and Pattern Recognition, Las Vegas, 2016: 1646-1654.
- Kim J, Lee J K, Lee K M. Deeply-recursive convolutional network for image
super-resolution[C]//Computer Vision and Pattern Recognition, Las Vegas, 2016: 1637-1645.
- Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European Conference on Computer Vision. Springer International Publishing, 2016.
- Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang, Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution, CVPR, 2017

