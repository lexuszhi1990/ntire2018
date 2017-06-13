function [label] = generate_training_h5(data_path)
  % usage:
  % addpath('./src/evaluation_mat');
  % addpath('./src/dataset_builder');
  % generate_training_h5('./dataset/train_291');

  f_lst = [];
  f_lst = [f_lst; dir(fullfile(data_path, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(data_path, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(data_path, '*.png'))];

  count = 0;
  patch_size = 256;
  label = zeros(patch_size, patch_size,1, 1, 'single');
  data_l2 = zeros(patch_size/2, patch_size/2,1, 1, 'single');
  data_l4 = zeros(patch_size/4, patch_size/4,1, 1, 'single');
  data_l8 = zeros(patch_size/8, patch_size/8,1, 1, 'single');

  %% writing to HDF5
  chunksz = 16;
  created_flag = false;
  totalct = 0;
  savepath = ['./dataset/train.h5'];

  for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end

    f_path = fullfile(data_path,f_info.name);
    img_raw = imread(f_path);

    % randomly resize between 0.5 ~ 1.0
    ratio = randi([5, 10]) * 0.1;
    img_raw = imresize(img_raw, ratio);

    % min width/height should be larger than patch size
    if( size(img_raw, 1) < patch_size || size(img_raw, 2) < patch_size )
      img_raw = vllab_imresize(img_raw, patch_size);
    end

    % random crop with size
    cropped_img = random_crop(img_raw, patch_size);

    % random rotate and flip
    image = random_rotate_and_flip(cropped_img);

    if size(img_raw, 3) == 3
      img_ycbcy = rgb2ycbcr(image);
    else
      disp(f_info);
      disp('only one channel for this image');
      continue;
    end

    img_y = img_ycbcy(:, :, 1);
    data_l2_y = imresize(img_y, 1/2, 'bicubic');
    data_l4_y = imresize(img_y, 1/4, 'bicubic');
    data_l8_y = imresize(img_y, 1/8, 'bicubic');

    count = count+1;
    label(:, :, :, count) = im2single(img_y);
    data_l2(:, :, :, count) = im2single(data_l2_y);
    data_l4(:, :, :, count) = im2single(data_l4_y);
    data_l8(:, :, :, count) = im2single(data_l8_y);

    disp(f_path);
  end

  order = randperm(count);
  label = label(:, :, :, order);
  data_l8 = data_l8(:, :, :, order);
  data_l4 = data_l4(:, :, :, order);
  data_l2 = data_l2(:, :, :, order);

  for batchno = 1:floor(count/chunksz)
      last_read=(batchno-1)*chunksz;
      batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
      batchdata_l2 = data_l2(:,:,:,last_read+1:last_read+chunksz);
      batchdata_l4 = data_l4(:,:,:,last_read+1:last_read+chunksz);
      batchdata_l8 = data_l8(:,:,:,last_read+1:last_read+chunksz);

      startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
      curr_dat_sz = store2hdf5_multipy(savepath, batchdata_l2, batchdata_l4, batchdata_l8, batchlabs, ~created_flag, startloc, chunksz);
      created_flag = true;
      totalct = curr_dat_sz(end);

      disp(['at batch: ', num2str(batchno)]);
  end

end