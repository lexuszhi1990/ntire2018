function [PSNR, SSIM, IFC] = eval_dataset(dataset_dir, method_dir, sr_method, sr_factor)

% usage:
% addpath('./src/evaluation_mat');
% addpath('./src/evaluation_mat/ifc-drrn');
% addpath('./src/evaluation_mat/matlabPyrTools');
% eval_dataset('./dataset/test/set14', 'lapsrn', 4)
% eval_dataset('./dataset/test/set5', 'lapsrn', 4)

dataset_gt_path = fullfile(dataset_dir, 'GT');
dataset_sr_path = fullfile(dataset_dir, method_dir);

gt_lst = dir(fullfile(dataset_gt_path ,'*.png'));
gt_lst = [gt_lst; dir(fullfile(dataset_gt_path, '*.bmp'))];

PSNR = zeros([1, numel(gt_lst)]);
SSIM = zeros([1, numel(gt_lst)]);
IFC = zeros([1, numel(gt_lst)]);

for n=1:numel(gt_lst)
  gt_img_name = gt_lst(n).name;
  gt_img_path = fullfile(dataset_gt_path, gt_img_name);

  gt_names = strsplit(gt_img_name, '.');
  generated_img_path = fullfile(dataset_sr_path, sprintf('%s_l%d_%s_x%d.png', gt_names{1}, sr_factor, sr_method, sr_factor)); % ./test/urban100/bicubic/img_100_bicubic_x8.png

  if ~exist(gt_img_path) | ~exist(generated_img_path)
    disp([gt_img_path, generated_img_path]);
    disp('img not exists...');
    return ;
  end

  % v2: DRRN
  gt_img = imread(gt_img_path);
  gt_img_ycbcr = rgb2ycbcr(gt_img);
  gt_img_y = im2double(gt_img_ycbcr(:,:,1));
  gt_img_y_shaved = shave(uint8(single(gt_img_y) * 255), [0, 0]);

  % generated_img = imread(generated_img_path);
  generated_img = imresize(imresize(gt_img, 1.0/sr_factor), sr_factor, 'bicubic');
  generated_img_ycbcr = rgb2ycbcr(generated_img);
  generated_img_y = im2double(generated_img_ycbcr(:,:,1));
  generated_img_y_shaved = shave(uint8(single(generated_img_y) * 255), [0, 0]);

  PSNR(n) = compute_psnr(gt_img, generated_img);
  SSIM(n) = compute_ssim(gt_img_y_shaved, generated_img_y_shaved);
  IFC(n) = ifcvec(double(gt_img_y_shaved), double(generated_img_y_shaved));

  fprintf('for image %s:\n--PSNR: %.4f;\tSSIM: %.4f;\tIFC: %.4f\n', generated_img_path, PSNR(n), SSIM(n), IFC(n));
end

fprintf('for dataset %s, upscaled by %s, at scale:%d\n--PSNR: %.4f;\tSSIM: %.4f;\tIFC: %.4f;\n', dataset_dir, sr_method, sr_factor, mean(PSNR), mean(SSIM), mean(IFC));

end
