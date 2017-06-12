function save_matrix(psnr, ssim, ifc, filename, images, precision)
% -------------------------------------------------------------------------
%   Description:
%       save a 2D array into text file
%
%   Input:
%       - psnr         : input matrix (2D array)
%       - ssim         : input matrix (2D array)
%       - ifc         : input matrix (2D array)
%       - filename  : output file name
%       - precision : float point precision [default = 7]
%
%   Citation:
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    if( ~exist('precision', 'var') )
        precision = 7;
    end

    file = fopen(filename, 'w');

    if( size(psnr, 2) == 1 )
        dlmwrite(filename, psnr, 'precision', precision, 'delimiter', '\n', 'newline', 'unix');
    else

        fprintf(file, '%15s:\t', 'results(matlab)');
        fprintf(file, sprintf('%%.%ds\t\t', precision), 'psnr');
        fprintf(file, sprintf('%%.%ds\t\t', precision), 'ssim');
        fprintf(file, sprintf('%%.%ds\t\n', precision), 'ifc');

        for i = 1:size(psnr, 2)
            fprintf(file, '%15s:\t', char(images(i, 1)));
            fprintf(file, sprintf('%%.%df\t', precision), psnr(1, i));
            fprintf(file, sprintf('%%.%df\t', precision), ssim(1, i));
            fprintf(file, sprintf('%%.%df\t\n', precision), ifc(1, i));
        end
        fprintf(file, '%15s:\t', 'mean');
        fprintf(file, sprintf('%%.%df\t', precision), mean(psnr(1, 1:end)));
        fprintf(file, sprintf('%%.%df\t', precision), mean(ssim(1, 1:end)));
        fprintf(file, sprintf('%%.%df\n', precision), mean(ifc(1, 1:end)));
        fprintf(file, '%15s:\t', 'std');
        fprintf(file, sprintf('%%.%df\t', precision), std(psnr(1, 1:end)));
        fprintf(file, sprintf('%%.%df\t', precision), std(ssim(1, 1:end)));
        fprintf(file, sprintf('%%.%df\n', precision), std(ifc(1, 1:end)));
        fprintf(file, '%15s:\t', 'var');
        fprintf(file, sprintf('%%.%df\t', precision), var(psnr(1, 1:end)));
        fprintf(file, sprintf('%%.%df\t', precision), var(ssim(1, 1:end)));
        fprintf(file, sprintf('%%.%df\n', precision), var(ifc(1, 1:end)));

    end

    fclose(file);
end
