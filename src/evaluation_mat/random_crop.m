function cropped_img = random_crop(img, patch_size)
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
  mask(1 + r1 : end - r2, 1 + r1 : end - r2) = 1;

  [X, Y] = meshgrid(1:W, 1:H);
  X = X(mask == 1);
  Y = Y(mask == 1);

  select = randperm(length(X), 1);
  X = X(select);
  Y = Y(select);

  cropped_img = img(Y - r1 : Y + r2, X - r1 : X + r2, :);

end
