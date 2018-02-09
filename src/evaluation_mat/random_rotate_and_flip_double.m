function [lr_img, hr_img] = random_rotate_and_flip_double(lr_input_img, hr_input_img)
% -------------------------------------------------------------------------
%   Description:
%       random rotate and flip
%
%   Input:
%       - input_img         : input image
% -------------------------------------------------------------------------

  hr_img = hr_input_img;
  lr_img = lr_input_img;

  % random rotate
  rotate = rand;
  if( rotate < 0.25 )
      lr_img = rot90(lr_img, 1);
      hr_img = rot90(hr_img, 1);
  elseif( rotate < 0.5 )
      lr_img = rot90(lr_img, 2);
      hr_img = rot90(hr_img, 2);
  elseif( rotate < 0.75 )
      lr_img = rot90(lr_img, 3);
      hr_img = rot90(hr_img, 3);
  end

  % horizontally flip
  if( rand > 0.5 )
      lr_img = fliplr(lr_img);
      hr_img = fliplr(hr_img);
  end

  % vertically flip
  if( rand > 0.5 )
      lr_img = flipud(lr_img);
      hr_img = flipud(hr_img);
  end
end
