function [] = generate_test_dataset(origin_dir, dataset_path)

  % usage:
  % addpath('./src/dataset_builder');
  % generate_test_dataset('./dataset/test/set14')

  f_lst = [];
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.jpg'))];
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.bmp'))];
  f_lst = [f_lst; dir(fullfile(origin_dir, '*.png'))];

  gt_img_dir = fullfile(dataset_path, 'GT');
  if ~exist(gt_img_dir)
    mkdir(gt_img_dir);
  end

  zoom_in_dir = fullfile(dataset_path, 'lr_x2348');
  if ~exist(zoom_in_dir)
    mkdir(zoom_in_dir);
  end

  bicubic_dir = fullfile(dataset_path, 'bicubic');
  if ~exist(bicubic_dir)
    mkdir(bicubic_dir);
  end

  scale_list = [2, 3, 4, 8];

  for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end

    f_path = fullfile(origin_dir,f_info.name);
    img_raw = imread(f_path);
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    img_raw = img_raw(1:height-mod(height,24),1:width-mod(width,24),:);
    % disp([mod(height,24), mod(width,24)]);

    patch_name = sprintf('%s.png',image_names{1});
    gt_img_path = fullfile(gt_img_dir, patch_name);
    imwrite(img_raw, gt_img_path);

    for i = 1:numel(scale_list)
      scale = scale_list(i);

      image_names = strsplit(f_info.name, '.');
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
