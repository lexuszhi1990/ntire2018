function psnr=psnr_index(im1,im2)

imdff = double(im1) - double(im2);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));
psnr = 20*log10(255/rmse);
