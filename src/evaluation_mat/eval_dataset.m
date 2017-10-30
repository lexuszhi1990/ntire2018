function [PSNR, SSIM, IFC] = eval_dataset(dataset_dir, sr_method, sr_factor)

%{
usage:
  addpath('./src/evaluation_mat');
  addpath('./src/evaluation_mat/ifc-drrn');
  addpath('./src/evaluation_mat/matlabPyrTools');

  eval_dataset('./dataset/mat_test/set5', 'LapSRN_v30', 2);
%}

dataset_gt_path = fullfile(dataset_dir, 'PNG');
dataset_sr_path = fullfile(dataset_dir, 'lapsrn', sr_method);

gt_lst = dir(fullfile(dataset_gt_path ,'*.png'));
gt_lst = [gt_lst; dir(fullfile(dataset_gt_path, '*.bmp'))];

PSNR = zeros([1, numel(gt_lst)]);
SSIM = zeros([1, numel(gt_lst)]);
IFC = zeros([1, numel(gt_lst)]);

image_names = [];

for n=1:numel(gt_lst)
  gt_img_name = gt_lst(n).name;
  gt_img_path = fullfile(dataset_gt_path, gt_img_name);

  gt_names = strsplit(gt_img_name, '.');
  generated_img_path = fullfile(dataset_sr_path, sprintf('%s_l%d_%s_x%d.png', gt_names{1}, sr_factor, sr_method, sr_factor)); % ./test/urban100/bicubic/img_100_bicubic_x8.png
  fprintf('for test image %s:\n', generated_img_path);

  if ~exist(gt_img_path) | ~exist(generated_img_path)
    disp([gt_img_path, generated_img_path]);
    disp('img not exists...');
    return ;
  end

  gt_img = imread(gt_img_path);
  gt_img_ycbcr = rgb2ycbcr(im2double(gt_img));
  gt_img_y = gt_img_ycbcr(:,:,1);
  gt_img_y = modcrop(gt_img_y, 24);
  gt_img_y_double = double(gt_img_y * 255);
  gt_img_y_shaved = shave(gt_img_y_double, sr_factor);

  generated_img = imread(generated_img_path);
  % generated_img = imresize(imresize(gt_img, 1.0/sr_factor), sr_factor, 'bicubic');
  generated_img_ycbcr = rgb2ycbcr(im2double(generated_img));
  generated_img_y = generated_img_ycbcr(:,:,1);
  generated_img_y_double = double(generated_img_y * 255);
  generated_img_y_shaved = shave(generated_img_y_double, sr_factor);

  PSNR(n) = compute_psnr(gt_img_y_shaved, generated_img_y_shaved);
  SSIM(n) = compute_ssim(gt_img_y_shaved, generated_img_y_shaved);
  IFC(n) = ifcvec(gt_img_y_double, generated_img_y_double);
  image_names = [image_names; {gt_img_name}];

  fprintf('--PSNR: %.4f;\tSSIM: %.4f;\tIFC: %.4f\n', PSNR(n), SSIM(n), IFC(n));
end

lapsrn_results=[[31.54 0.885 3.559]; [28.19 0.728 3.147];[27.32 0.728 2.677]];
dataset_index=1;
pos = strfind(dataset_dir, 'set14');
if (~isempty(pos))
  dataset_index=2;
end
pos = strfind(dataset_dir, 'bsd100');
if (~isempty(pos))
  dataset_index=3;
end

fprintf('\nfor dataset %s, upscaled by %s, at scale:%d\n--Average PSNR/SSIM/IFC: \t %.4f/%.4f/%.4f\n Average improvements PSNR/SSIM/IFC: %.4f/%.4f/%.4f\n ', dataset_dir, sr_method, sr_factor, mean(PSNR), mean(SSIM), mean(IFC), mean(PSNR)-lapsrn_results(dataset_index, 1), mean(SSIM)-lapsrn_results(dataset_index, 2), mean(IFC)-lapsrn_results(dataset_index, 3));

filename = fullfile(dataset_sr_path, ['results-finally-' sr_method '-' num2str(sr_factor) '.txt']);
save_matrix(PSNR, SSIM, IFC, filename, image_names);
fprintf('save result at %s\n', filename);
