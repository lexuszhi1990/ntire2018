function psnr=compute_psnr(im1,im2)
if size(im1, 3) == 3,
    im1 = rgb2ycbcr(im1);
    im1 = im1(:, :, 1);
end

if size(im2, 3) == 3,
    im2 = rgb2ycbcr(im2);
    im2 = im2(:, :, 1);
end

psnr = psnr_index(im1, im2);
