function get_ycbcr_image(image_path, mat_path, scale)
    img = imread(image_path);
    img_raw = im2double(img);
    img_ycbcr = rgb2ycbcr(img_raw);
    img_y = img_ycbcr(:,:,1);
    bicubic_img_ycbcr = imresize(img_ycbcr, scale, 'bicubic');
    save(mat_path, 'img_raw', 'img_ycbcr', 'bicubic_img_ycbcr', 'img_y');
