function [data, label] = generate_combined_dataset(data_path, dataset)
  % addpath('./src/dataset_builder');

  % generate_combined_dataset('./dataset', '291_origin')

  dataDir = fullfile(data_path, dataset);
  f_lst = [];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];

  count = 0;
  default_width = 480;
  default_height = 320;
  data = zeros(default_height, default_width, 3, 1);
  label = zeros(default_height, default_width,3, 1);
  %% writing to HDF5
  chunksz = 10;
  created_flag = false;
  totalct = 0;
  savepath = ['./dataset/train_dataset_' dataset '.h5'];

  for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end

    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    if width < height
      img_raw = imrotate(img_raw, 90);
      img_size = size(img_raw);
      width = img_size(2);
      height = img_size(1);
    end

    if width < default_width/3 | height < default_height/3
      disp(img_size);
      disp('this image is too small...\n');
      continue;
    end

    % f_size = floor(min(width, height)/2);
    if height * 1.5 > width
      y_size = floor(width / 3);
      x_size = floor(width / 2);
    else
      x_size = floor(height * 3 / 4);
      y_size = floor(height / 2);
    end
    center_width=round(width/2);
    center_height=round(height/2);
    BoxBottomX=center_width-x_size;
    BoxBottomY=center_height-y_size;
    BoxTopX=center_width+x_size;
    BoxTopY=center_height+y_size;
    croped_img = imcrop(img_raw, [BoxBottomX BoxBottomY BoxTopX BoxTopY]);
    gt_img = imresize(croped_img, [default_height default_width] ,'bicubic');

    % gt_img=im2double(gt_img);
    gt_img=im2uint8(gt_img);

    if img_size(3) == 3
      img_ycbcy = rgb2ycbcr(gt_img);
    else
      disp(f_info);
      disp('only one channel for this image');
      continue;
    end

    count=count+1;
    data(:, :, :, count) = single(img_ycbcy);
    label(:, :, :, count) = single(gt_img);

    image_names = strsplit(f_info.name, '.');
    image_name = image_names{1};
    patch_name = sprintf('%s.png' ,image_name);
    png_img_path = fullfile('./dataset/291_png', patch_name);
    imwrite(gt_img, png_img_path);
    disp(png_img_path);
  end

  order = randperm(count);
  data = data(:, :, :, order);
  label = label(:, :, :, order);

  for batchno = 1:floor(count/chunksz)
      last_read=(batchno-1)*chunksz;
      batchdata = data(:,:,:,last_read+1:last_read+chunksz);
      batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

      startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
      curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
      created_flag = true;
      totalct = curr_dat_sz(end);

      disp(batchno);
  end
  % h5disp(savepath);
end
