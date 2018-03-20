function [label] = generate_training_h5(HR_path, LR_path, epoches, saved_name)
%{
  usage:
    addpath('./src/evaluation_mat');
    addpath('./src/dataset_builder');
    generate_training_h5('../dataset/DIV2K_train_HR', '../dataset/DIV2K_train_LR_difficult', 150, 'div2k_x');

  dev:
    HR_path = '../dataset/DIV2K_train_HR';
    LR_path = '../dataset/DIV2K_train_LR_difficult';

    f_lst = [];
    f_lst = [f_lst; dir(fullfile(HR_path, '*.png'))];
    f_info = f_lst(1);
    f_path = fullfile(HR_path,f_info.name);
    hr_img_raw = imread(f_path);

    split_names = strsplit(f_info.name, '.');
    lr_img_path = strcat(LR_path, '/', char(split_names(1)), 'x4d.png');
    lr_img_raw = imread(lr_img_path);


    patch_size = 64;
    [lr_img_patch, hr_img_patch] = random_crop_x4(lr_img_raw, hr_img_raw, patch_size);
    [lr_img, hr_img] = random_rotate_and_flip_double(lr_img_patch, hr_img_patch);

    lr_img = im2double(lr_img);
    lr_img_ycbcy = rgb2ycbcr(lr_img);

    hr_img = im2double(hr_img);
    hr_img_ycbcy = rgb2ycbcr(hr_img);

    hr_img_y = hr_img_ycbcy(:, :, 1);
    lr_img_y = lr_img_ycbcy(:, :, 1);

    data_l2_y = imresize(hr_img_y, 1/2, 'bicubic');
    data_l4_y = imresize(hr_img_y, 1/4, 'bicubic');
    data_l8_y = imresize(hr_img_y, 1/8, 'bicubic');

    count = count+1;
    label(:, :, :, count) = hr_img_y;
    data_l2(:, :, :, count) = data_l2_y;
    data_l4(:, :, :, count) = lr_img_y;
    data_l8(:, :, :, count) = data_l8_y;

%}

  f_lst = [];
  f_lst = [f_lst; dir(fullfile(HR_path, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(HR_path, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(HR_path, '*.png'))];

  if( ~exist('epoches', 'var') )
      epoches = 1;
  end
  if( ~exist('saved_name', 'var') )
    disp('please input var name');
    return
  end

  count = 0;
  patch_size = 96;
  label = zeros(patch_size*4, patch_size*4,1, 1, 'single');
  data_l2 = zeros(patch_size*2, patch_size*2,1, 1, 'single');
  data_l4 = zeros(patch_size, patch_size,1, 1, 'single');
  data_l8 = zeros(patch_size/2, patch_size/2,1, 1, 'single');

  %% writing to HDF5
  chunksz = 16;
  created_flag = false;
  totalct = 0;
  savepath = ['./dataset/' saved_name  num2str(epoches) '.h5'];
  disp(savepath);

  for epoch = 1:epoches
    f_lst = f_lst(randperm(length(f_lst)));
    for f_iter = 1:numel(f_lst)
      f_info = f_lst(f_iter);
      if f_info.name == '.'
          continue;
      end

      f_path = fullfile(HR_path,f_info.name);
      hr_img_raw = imread(f_path);

      split_names = strsplit(f_info.name, '.');
      lr_img_path = strcat(LR_path, '/', char(split_names(1)), 'x4d.png');
      lr_img_raw = imread(lr_img_path);

      [lr_img_patch, hr_img_patch] = random_crop_x4(lr_img_raw, hr_img_raw, patch_size);
      [lr_img, hr_img] = random_rotate_and_flip_double(lr_img_patch, hr_img_patch);

      lr_img = im2double(lr_img);
      lr_img_ycbcy = rgb2ycbcr(lr_img);

      hr_img = im2double(hr_img);
      hr_img_ycbcy = rgb2ycbcr(hr_img);

      hr_img_y = hr_img_ycbcy(:, :, 1);
      lr_img_y = lr_img_ycbcy(:, :, 1);

      data_l2_y = imresize(hr_img_y, 1/2, 'bicubic');
      data_l4_y = imresize(hr_img_y, 1/4, 'bicubic');
      data_l8_y = imresize(hr_img_y, 1/8, 'bicubic');

      count = count+1;
      label(:, :, :, count) = hr_img_y;
      data_l2(:, :, :, count) = data_l2_y;
      data_l4(:, :, :, count) = lr_img_y;
      data_l8(:, :, :, count) = data_l8_y;
    end

    disp([num2str(epoch) 'end...']);
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
      curr_dat_sz = store2hdf5_multipy_x4(savepath, batchdata_l2, batchdata_l4, batchdata_l8, batchlabs, ~created_flag, startloc, chunksz);
      created_flag = true;
      totalct = curr_dat_sz(end);

      disp(['at batch: ', num2str(batchno)]);
  end

end
