function [PSNR, SSIM, IFC] = eval_dataset_mat(dataset_dir, method_dir, sr_method, sr_factor)

  %{
  usage:
    addpath('./src/evaluation_mat');
    addpath('./src/evaluation_mat/ifc-drrn');
    addpath('./src/evaluation_mat/matlabPyrTools');

    eval_dataset_mat('./dataset/mat_test/set5', 'lapsrn/mat', 'LapSRN_v1', 4);
  %}

  dataset_gt_path = fullfile(dataset_dir, 'mat');
  dataset_sr_path = fullfile(dataset_dir, method_dir);
  if ~exist(dataset_sr_path)
    mkdir(dataset_sr_path);
  end

  gt_lst = dir(fullfile(dataset_gt_path ,'*.mat'));

  PSNR = zeros([1, numel(gt_lst)]);
  SSIM = zeros([1, numel(gt_lst)]);
  IFC = zeros([1, numel(gt_lst)]);

  image_names = [];

  for n=1:numel(gt_lst)
    gt_img_name = gt_lst(n).name;
    gt_names = strsplit(gt_img_name, '.');

    gt_img_path = fullfile(dataset_gt_path, gt_img_name);
    img_mat = load(gt_img_path);

    gt_img_y = getfield(img_mat, sprintf('label_x8_y'));
    im_l_ycbcr = getfield(img_mat, sprintf('label_x%d_ycbcr', 8/sr_factor));
    im_h_ycbcr = imresize(im_l_ycbcr, sr_factor, 'bicubic');
    upscaled_img_y = getfield(img_mat, sprintf('%s_l%d_x%d_y', sr_method, sr_factor, sr_factor));

    gt_img_y_uint8 = im2uint8(gt_img_y);
    im_h_ycbcr_uint8 = im2uint8(im_h_ycbcr);
    upscaled_img_y_uint8 = im2uint8(upscaled_img_y);

    gt_img_y_shaved = shave(gt_img_y_uint8, sr_factor);
    upscaled_img_y_shaved = shave(upscaled_img_y_uint8, sr_factor);

    PSNR(n) = compute_psnr(gt_img_y_shaved, upscaled_img_y_shaved);
    SSIM(n) = compute_ssim(gt_img_y_shaved, upscaled_img_y_shaved);
    IFC(n) = compute_ifc(double(gt_img_y_shaved), double(upscaled_img_y_shaved));
    image_names = [image_names; {gt_img_name}];

    fprintf('--PSNR: %.4f;\tSSIM: %.4f;\tIFC: %.4f\n', PSNR(n), SSIM(n), IFC(n));

    im_h_ycbcr(:,:,1) = upscaled_img_y;
    upscaled_img = ycbcr2rgb(im_h_ycbcr);
    upscaled_img_path = fullfile(dataset_sr_path, sprintf('%s_l%d_%s_x%d.png', gt_names{1}, sr_factor, sr_method, sr_factor));
    imwrite(upscaled_img, upscaled_img_path);
    fprintf('save image at %s\n', upscaled_img_path);

  end

  fprintf('\nfor dataset %s, upscaled by %s, at scale:%d\n--Average PSNR/SSIM/IFC: \t %.4f/%.4f/%.4f\n', dataset_dir, sr_method, sr_factor, mean(PSNR), mean(SSIM), mean(IFC));

  filename = fullfile(dataset_sr_path, ['results-' sr_method '-' num2str(sr_factor) '.txt']);
  save_matrix(PSNR, SSIM, IFC, filename, image_names);
  fprintf('save result at %s\n', filename);

end
