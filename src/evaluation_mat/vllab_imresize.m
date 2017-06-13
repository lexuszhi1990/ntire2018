function img = vllab_imresize(img, target_size)
% -------------------------------------------------------------------------
%   Description:
%       resize the minimal side to target_size
%
%   Input:
%       - img         : input image
%       - target_size : target image size (min width/height)
%
%   Output:
%       - img   : output image
% -------------------------------------------------------------------------

    H = size(img, 1);
    W = size(img, 2);

    if( H < W )
        ratio = target_size / H;
    else
        ratio = target_size / W;
    end

    img = imresize(img, ratio);

end
