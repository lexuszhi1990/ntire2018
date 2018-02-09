function [cropped_lr_img, cropped_hr_img] = random_crop_x4(img, hr_img, patch_size)
% -------------------------------------------------------------------------
%   Description:
%       resize the minimal side to patch_size
%
%   Input:
%       - img         : input image
%       - patch_size : target image size (min width/height)
% -------------------------------------------------------------------------

  img_size = size(img);
  H = img_size(1);
  W = img_size(2);

  r1 = floor(patch_size / 2);
  r2 = patch_size - r1 - 1;
  mask = zeros(H, W);
  mask(4 + r1 : end - r2 - 4, 4 + r1 : end - r2 - 4) = 1;

  [X, Y] = meshgrid(1:W, 1:H);
  X = X(mask == 1);
  Y = Y(mask == 1);

  select = randperm(length(X), 1);
  X = X(select);
  Y = Y(select);

  cropped_lr_img = img(Y - r1 : Y + r2, X - r1 : X + r2, :);
  disp([Y - r1, Y + r2, X - r1, X + r2]);
  disp([ceil((Y - r1)*4), ceil((Y + r2+1)*4), ceil((X - r1)*4), ceil((X + r2+1)*4)]);
  cropped_hr_img = hr_img(ceil((Y - r1-0.5)*4) : floor((Y + r2+0.5)*4)-1, ceil((X - r1-0.5)*4) : floor((X + r2+0.5)*4)-1, :);
  % disp([size(cropped_lr_img) size(cropped_hr_img)]);

end
