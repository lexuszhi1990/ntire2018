function sr_img_raw = save_ycbcr_image(mat_dir, saved_dir, bicubic_dir)
    img_mat = load(mat_dir);
    gt_img_y = getfield(img_mat, sprintf('sr_img_y'));
    sr_img = getfield(img_mat, sprintf('bicubic_img_ycbcr'));
    % imwrite(ycbcr2rgb(sr_img), bicubic_dir);
    sr_img(:,:,1) = gt_img_y(:,:,1);
    sr_img_raw = ycbcr2rgb(sr_img);
    imwrite(sr_img_raw, saved_dir);
