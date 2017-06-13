% update train dataset
% usage:
% os.system('matlab -nodesktop -nosplash -r train_h5_eval');

addpath('./src/evaluation_mat');
addpath('./src/dataset_builder');

generate_training_h5('./dataset/train_291');

exit;
