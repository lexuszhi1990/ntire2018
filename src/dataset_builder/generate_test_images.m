function [] = generate_test_images(dataset_path)

  % usage:
  % addpath('./src/dataset_builder');
  % generate_test_images('./dataset/test/set14')

  scale_list = [2, 3, 4, 8];

  f_lst = [];
  origin_dir = fullfile(dataset_path, 'GT');
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.png'))];

  mat_dir = fullfile(dataset_path, 'mat');
  if ~exist(mat_dir)
    mkdir(mat_dir);
  end

  zoom_in_dir = fullfile(dataset_path, 'lr_x2348');
  if ~exist(zoom_in_dir)
    mkdir(zoom_in_dir);
  end

  bicubic_dir = fullfile(dataset_path, 'bicubic');
  if ~exist(bicubic_dir)
    mkdir(bicubic_dir);
  end

  gt_img_dir = fullfile(dataset_path, 'PNG');
  if ~exist(gt_img_dir)
    mkdir(gt_img_dir);
  end

  for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end

    image_names = strsplit(f_info.name, '.');

    f_path = fullfile(origin_dir ,f_info.name);
    img = imread(f_path);
    img_size = size(img);
    height = img_size(1); % 行数
    width = img_size(2); % 列数
    img_rgb = img(1:height-mod(height,24),1:width-mod(width,24),:);

    if size(img, 3) == 3
      img_raw = im2double(img_rgb);
    else
      disp(['only one channel for this image ' f_info.name]);
      img_raw = im2double(repmat(img_rgb, [1 1 3]));
    end

    % save mat file for each image
    label_x8 = img_raw;
    label_x8_ycbcr = rgb2ycbcr(label_x8);
    label_x8_y = label_x8_ycbcr(:,:,1);

    label_x4 = imresize(img_raw, 1/2,'bicubic');
    label_x4_ycbcr = rgb2ycbcr(label_x4);
    label_x4_y = label_x4_ycbcr(:,:,1);

    label_x2 = imresize(img_raw, 1/4,'bicubic');
    label_x2_ycbcr = rgb2ycbcr(label_x2);
    label_x2_y = label_x2_ycbcr(:,:,1);

    data = imresize(img_raw, 1/8,'bicubic');
    data_ycbcr = rgb2ycbcr(data);
    data_y = data_ycbcr(:,:,1);

    lm_path = fullfile(mat_dir, image_names{1});
    save(lm_path, 'img_rgb', 'label_x8_ycbcr', 'label_x8_y', 'label_x4_ycbcr', 'label_x4_y', 'label_x2_ycbcr', 'label_x2_y', 'data_ycbcr', 'data_y');
    disp(['save image ' lm_path]);

    patch_name = sprintf('%s.png',image_names{1});
    gt_img_path = fullfile(gt_img_dir, patch_name);
    imwrite(img_rgb, gt_img_path);

    for i = 1:numel(scale_list)
      scale = scale_list(i);

      patch_name = sprintf('%s_l%d.png',image_names{1}, scale);
      lm_path = fullfile(zoom_in_dir, patch_name);
      lr_img = imresize(img_raw,1/scale,'bicubic');
      imwrite(lr_img, lm_path);
      disp(lm_path);

      patch_name = sprintf('%s_l%d_bicubic_x%d.png',image_names{1}, scale, scale);
      bicubic_hr_path = fullfile(bicubic_dir, patch_name);
      bicubic_upscaled = imresize(lr_img, scale,'bicubic');
      imwrite(bicubic_upscaled, bicubic_hr_path);
      disp(bicubic_hr_path);
    end
  end
