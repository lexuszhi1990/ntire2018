function [PSNR, SSIM, IFC] = eval_dataset(dataset_dir, sr_method, sr_factor)

% usage:
% eval_dataset('../../dataset/test/set5', 'lapsrn', 4)

addpath('./matlabPyrTools/');
addpath('./IFC/');

dataset_gt_path = fullfile(dataset_dir, 'GT');
dataset_sr_path = fullfile(dataset_dir, sr_method);

gt_lst = dir(fullfile(dataset_gt_path ,'*.png'));
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

  gt_img = imread(gt_img_path);
  generated_img = imread(generated_img_path);

  if size(gt_img,3) == 1
    gt_img = cat(3, gt_img, gt_img, gt_img);
  end
  gt_img_ycbcr = rgb2ycbcr(im2double(gt_img));
  gt_img_y = gt_img_ycbcr(:,:,1);
  gt_img_y_shaved = shave(gt_img_y, [sr_factor, sr_factor]);

  if size(generated_img,3) == 1
    generated_img = cat(3, generated_img, generated_img, generated_img);
  end
  generated_img_ycbcr = rgb2ycbcr(im2double(generated_img));
  generated_img_y = generated_img_ycbcr(:,:,1);
  generated_img_y_shaved = shave(generated_img_y, [sr_factor, sr_factor]);

  PSNR(n) = psnr_index(gt_img_y_shaved, generated_img_y_shaved);
  SSIM(n) = ssim_index(gt_img_y_shaved, generated_img_y_shaved);
  IFC(n) = ifcvec(double(gt_img_y_shaved), double(generated_img_y_shaved));

  fprintf('for image %s:\n--PSNR: %.4f\tSSIM: %.4f\tIFC: %.4f\n', gt_img_name, PSNR(n), SSIM(n), IFC(n));
end

fprintf('for dataset %s, upscaled by %s, at scale:%d\n--PSNR: %.4f\tSSIM: %.4f\tIFC: %.4f\n', dataset_dir, sr_method, sr_factor, mean(PSNR), mean(SSIM), mean(IFC));

end
