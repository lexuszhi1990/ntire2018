function [data, label] = generate_training_dataset(data_path, dataset)
  % usage:
  % generate_training_dataset('./dataset/test/set5', 'GT')

  dataDir = fullfile(data_path, dataset);
  f_lst = [];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];

  count = 0;
  default_width = 312;
  default_height = 312;
  data = zeros(default_width, default_height, 3, 1);
  label = zeros(default_width, default_height,1, 1);
  %% writing to HDF5
  chunksz = 16;
  created_flag = false;
  totalct = 0;
  savepath = ['./train_dataset_' dataset '.h5'];

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

    % image_names = strsplit(f_info.name, '.');
    % image_name = image_names{1};
    % patch_name = sprintf('%s/%s',folder,image_name);
    % disp(patch_name);

    % img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    % disp(img_size)
    if width < default_width | height < default_height
      disp(img_size);
      continue;
    end
    f_size = floor(min(width, height)/2);
    center_width=round(width/2);
    center_height=round(height/2);
    BoxBottomX=center_width-f_size;
    BoxBottomY=center_height-f_size;
    BoxTopX=center_width+f_size;
    BoxTopY=center_height+f_size;
    croped_img = imcrop(img_raw, [BoxBottomX BoxBottomY BoxTopX BoxTopY]);
    gt_img = imresize(croped_img, [default_width default_height] ,'bicubic');

    if img_size(3) == 3
      img_ycbcy = rgb2ycbcr(gt_img);
      yc_img = img_ycbcy(:,:,1);
    else
      disp(f_info);
      disp('only one channel for this image');
      % img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
      return;
    end

    % 0 degree
    count=count+1;
    gt_img_r0 = imrotate(gt_img, 0);
    data(:, :, :, count) = single(gt_img_r0);
    y_img_r0 = imrotate(yc_img, 0);
    label(:, :, :, count) = single(y_img_r0);

    % 90 degree
    count=count+1;
    gt_img_r180 = imrotate(gt_img, 180);
    data(:, :, :, count) = single(gt_img_r180);
    y_img_r180 = imrotate(yc_img, 180);
    label(:, :, :, count) = single(y_img_r180);

    disp(f_path);
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

  return;
