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

references:
https://cn.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

### demo

CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 --batch_size=1 --channel=1 --filter_num=64 --sr_method=EDSR_v315 --model_path=./ckpt/EDSR_v315/

### train
