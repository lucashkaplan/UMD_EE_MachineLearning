%% ENEE436 Project 2
% Author: Lucas Kaplan

% Goal: Classify handwritten digits using multi-class linear SVM.

% This implementation uses the MNIST dataset and the LIBSVM library, both
% of which are available online.

%% Loading Data
clc; clear vars;

% compiling LIBSVM
cd libsvm-3.31\matlab\;
fprintf('MEX... \n');
make;
fprintf('\n');
cd ../../;

% add path to use svmtrain function
addpath('libsvm-3.31\matlab\')

% load data
load('mnist.mat')

trainImgs = training.images;
% reshape images to 2D matrix: (dimensions) x (no. of imgs)
trainImgs = reshape(trainImgs, size(trainImgs, 1) * size(trainImgs, 2), size(trainImgs, 3));
% transpose s.t. images stored as: (no. of imgs) x (dimensions)
trainImgs = trainImgs';

trainLabels = training.labels;

testImgs = test.images;
% reshape images to 2D matrix: (dimensions) x (no. of imgs)
testImgs = reshape(testImgs, size(testImgs, 1) * size(testImgs, 2), size(testImgs, 3));
% transpose s.t. images stored as: (no. of imgs) x (dimensions)
testImgs = testImgs';

testLabels = test.labels;

%% PCA
% desired number of reduced dimensions
reducedDim = 150;

% training imgs
eigenvec_matrix = pca(trainImgs);
trainImgs = trainImgs * (eigenvec_matrix(:, 1:reducedDim));

% testing imgs
eigenvec_matrix = pca(testImgs);
testImgs = testImgs * (eigenvec_matrix(:, 1:reducedDim));

%% Training and Testing
tic;

% set options for SVM
% -t 0 = linear kernels
options = '-t 0 -q';
% train model on train data
model = svmtrain(trainLabels, trainImgs, options);
% compute accuracy on test data
[predictedLabels, accuracy, prob_estimates] = svmpredict(testLabels, testImgs, model);

fprintf('Time: %0.2f mins \n', toc/60);
