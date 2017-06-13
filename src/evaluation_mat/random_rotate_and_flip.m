function image = random_rotate_and_flip(input_img)
% -------------------------------------------------------------------------
%   Description:
%       random rotate and flip
%
%   Input:
%       - input_img         : input image
% -------------------------------------------------------------------------

  image = input_img;

  % random rotate
  rotate = rand;
  if( rotate < 0.25 )
      image = rot90(image, 1);
  elseif( rotate < 0.5 )
      image = rot90(image, 2);
  elseif( rotate < 0.75 )
      image = rot90(image, 3);
  end

  % horizontally flip
  if( rand > 0.5 )
      image = fliplr(image);
  end

  % vertically flip
  if( rand > 0.5 )
      image = flipud(image);
  end
end
