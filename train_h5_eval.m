% update train dataset
% usage:
% os.system('matlab -nodesktop -nosplash -r train_h5_eval');

addpath('./src/evaluation_mat');
addpath('./src/dataset_builder');

% generate_training_h5('./dataset/train_291_coco_347', 2, 'train_291_coco_347_x');
generate_training_h5('./dataset/train_291', 10, 'train_x');

exit;
