% get test image

function [img_ycbcr] = get_img(f_path)

    img = imread(f_path);

    if size(img, 3) == 3
      img_raw = im2double(img);
    else
      disp(['only one channel for this image ' f_info.name]);
      img_raw = im2double(repmat(img, [1 1 3]));
    end

    img_ycbcr = rgb2ycbcr(img_raw);

end
