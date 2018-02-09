import h5py

hf = h5py.File('./dataset/div2k_x5.h5')
label_x8 = hf.get("label_x8")
label_x4 = hf.get("label_x4")
label_x2 = hf.get("label_x2")
data = hf.get("data")
