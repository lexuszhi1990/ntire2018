lap_srn implement based tf

### dependency

```
pip install ipython --user
pip install h5py --user
sudo apt-get install libopencv-dev python-opencv
```


`git clone https://github.com/LabForComputationalVision/matlabPyrTools.git ./src/evaluation_mat/matlabPyrTools`


### roadmap

####2017-6-14

lapsrn lap_pry_x4_small.h5

./dataset/test/set5 scale:4
--Average PSNR: 30.1862; SSIM: 0.8635; IFC: 2.8144;

####2017-6-15

lapsrn train_x2.h5

python train.py --dataset_dir=./dataset/train_x2.h5 --continued_training=False --g_decay_rate=0.5 --gpu_id=2 --epoches=100 --lr=0.0001 --batch_size=8

for dataset ./dataset/test/set5, upscaled by lapsrn, at scale:4
--Average PSNR: 30.5380;        Average SSIM: 0.8683;   Average IFC: 2.9960;

python train.py --dataset_dir=./dataset/train_x5.h5 --continued_training=False --g_decay_rate=0.5 --gpu_id=2 --epoches=20 --lr=0.0001 --batch_size=8

for dataset ./dataset/test/set5, upscaled by lapsrn, at scale:4
--Average PSNR: 30.8876;        Average SSIM: 0.8764;   Average IFC: 3.1621;

for dataset ./dataset/test/set14, upscaled by lapsrn, at scale:4
--Average PSNR: 27.8565;        Average SSIM: 0.7639;   Average IFC: 2.8467;

### 2017-6-19

python val.py --gpu_id=0 --channel=1 --model=./ckpt/lapsrn/lapsrn-epoch-60-step-327-2017-06-19-22-25.ckpt-327 --image=./dataset/test/set5/mat --scale=8

It takes 0.213274002075s for processing
for dataset ./dataset/test/set5/mat, scale: 2, exec time: 0.2907s
--PSNR: 36.5536;        SSIM: 0.9554

for dataset ./dataset/test/set5/mat, scale: 4, exec time: 0.1112s
--PSNR: 30.9073;        SSIM: 0.8799

for dataset ./dataset/test/set5/mat, scale: 8, exec time: 0.1009s
--PSNR: 25.7404;        SSIM: 0.7324

python val.py --gpu_id=0 --channel=1 --model=./ckpt/lapsrn/lapsrn-epoch-60-step-327-2017-06-19-22-25.ckpt-327 --image=./dataset/test/set14/mat --scale=8
It takes 0.17630982399s for processing
for dataset ./dataset/test/set14/mat, scale: 2, exec time: 0.2785s
--PSNR: 32.2562;        SSIM: 0.9065

for dataset ./dataset/test/set14/mat, scale: 4, exec time: 0.1781s
--PSNR: 27.5943;        SSIM: 0.7676

for dataset ./dataset/test/set14/mat, scale: 8, exec time: 0.1772s
--PSNR: 24.0746;        SSIM: 0.6223

### 2017-6-19 31.0016 drrn

python val.py --gpu_id=0 --channel=1 --filter_num=128 --sr_method=lapsrn_drrn --model=./ckpt/lapsrn-solver_v2/lapsrn-epoch-4-step-181-2017-06-25-04-04.ckpt-181 --image=./dataset/test/set5/mat --scale=4

It takes 1.14187383652s for processing
save mat at ./dataset/test/set5/mat/baby_GT.mat in lapsrn_l4_x4_y
for dataset ./dataset/test/set5/mat, scale: 2, average exec time: 0.6681s
--Aaverage PSNR: 36.3319;       Aaverage SSIM: 0.9529

for dataset ./dataset/test/set5/mat, scale: 4, average exec time: 0.7015s
--Aaverage PSNR: 30.9160;       Aaverage SSIM: 0.8780


### 2017-07-08

for dataset ./dataset/test/set5, upscaled by lapsrn_mat_v2, at scale:4
--Average PSNR: 31.1038;        Average SSIM: 0.8818;   Average IFC: 3.2382;


### speed table

set5:
  SRCNN: 0.5343s
  VDSR(pytorch): 0.1068s
  DRCN(tensorflow):  0.65747s
  SubPixel(tf): 0.1191s
  LapSRN(tf): 0.1084s
  ESCN(v7): 0.8104s

set14:
  DRCN(tensorflow):  0.65382s


### set5 resolution
$ identify baby_GT_l4.png
baby_GT_l4.png PNG 128x128 128x128+0+0 8-bit sRGB 31.1KB 0.000u 0:00.000

$ identify bird_GT_l4.png
bird_GT_l4.png PNG 72x72 72x72+0+0 8-bit sRGB 11.3KB 0.000u 0:00.000

$ identify butterfly_GT_l4.png
butterfly_GT_l4.png PNG 64x64 64x64+0+0 8-bit sRGB 10.4KB 0.000u 0:00.000

$ identify head_GT_l4.png
head_GT_l4.png PNG 70x70 70x70+0+0 8-bit sRGB 9.05KB 0.000u 0:00.000

$ identify woman_GT_l4.png
woman_GT_l4.png PNG 57x86 57x86+0+0 8-bit sRGB 10.8KB 0.000u 0:00.000
