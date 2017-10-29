function [label] = generate_training_h5(data_path, epoches, saved_name)
%{
  usage:
    addpath('./src/evaluation_mat');
    addpath('./src/dataset_builder');
<<<<<<< HEAD
    generate_training_h5('./dataset/train', 1, 'LFW_SR_');
=======
    generate_training_h5('/home/mcc207/datasets/291', 5, "mat_train_391_x");
>>>>>>> f32f3e336de46642e0f5e2ff553cf1211cb384e6
%}

  f_lst = [];
  f_lst = [f_lst; dir(fullfile(data_path, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(data_path, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(data_path, '*.png'))];

  if( ~exist('epoches', 'var') )
      epoches = 1;
  end
  if( ~exist('saved_name', 'var') )
    disp('please input var name');
    return
  end

  count = 0;
  patch_size = 48;
  label = zeros(patch_size, patch_size,1, 1, 'single');
  data_l2 = zeros(patch_size/2, patch_size/2,1, 1, 'single');
  data_l4 = zeros(patch_size/4, patch_size/4,1, 1, 'single');
  data_l8 = zeros(patch_size/8, patch_size/8,1, 1, 'single');

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

      f_path = fullfile(data_path,f_info.name);
      img_raw = imread(f_path);
      % disp(size(img_raw));

      % randomly resize between 0.5 ~ 1.0
      ratio = randi([7, 10]) * 0.1;
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
        image = image;
      else
        disp(f_info);
        disp('only one channel for this image');
        continue;
      end

      image = im2double(image);
      img_ycbcy = rgb2ycbcr(image);

      img_y = img_ycbcy(:, :, 1);
      data_l2_y = imresize(img_y, 1/2, 'bicubic');
      data_l4_y = imresize(img_y, 1/4, 'bicubic');
      data_l8_y = imresize(img_y, 1/8, 'bicubic');

      count = count+1;
      label(:, :, :, count) = img_y;
      data_l2(:, :, :, count) = data_l2_y;
      data_l4(:, :, :, count) = data_l4_y;
      data_l8(:, :, :, count) = data_l8_y;

      disp(f_path);
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
      curr_dat_sz = store2hdf5_multipy(savepath, batchdata_l2, batchdata_l4, batchdata_l8, batchlabs, ~created_flag, startloc, chunksz);
      created_flag = true;
      totalct = curr_dat_sz(end);

      disp(['at batch: ', num2str(batchno)]);
  end

end
