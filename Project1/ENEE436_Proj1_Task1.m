%% ENEE436 Project 1 - Task 1
% Author: Lucas Kaplan

% Goal: Classify the subject of a test image.

% This implementation uses the POSE dataset, which contains 68 subjects
% under 13 different poses.

%% Load Pose Data
% Dataset
data_folder = './Data/';

% Test Ratio
test_ratio = 0.2;

load([data_folder,'pose.mat'])
[rows,columns,images,subjects]= size(pose);

% Convert the datase in data vectors and labels for subject identification
data = [];
labels = [];
for s=1:subjects
    for i=1:images
        pose_vector = reshape(pose(:,:,i,s),1,rows*columns);
        data = [data;pose_vector];
        labels = [labels s];        
    end
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
pose_train_data = data(idx(1:N),:);
pose_train_labels = labels(idx(1:N));
pose_test_data = data(idx(N+1:data_len),:);
pose_test_labels = labels(idx(N+1:data_len));

% make labels go from 0 to 67
pose_train_labels = pose_train_labels - 1;
pose_test_labels = pose_test_labels - 1;

%% PCA
% desired number of reduced dimensions
reducedDim = 300;

eigenvec_matrix = pca(pose_train_data);
Rpose_train_data = pose_train_data * (eigenvec_matrix(:, 1:reducedDim));
Rpose_test_data = pose_test_data * (eigenvec_matrix(:, 1:reducedDim));

%% Bayes Classifier
pose_classified_data = bayesClassifier(Rpose_train_data, pose_train_labels, Rpose_test_data, 'pose');
accuracy = classifierAccuracy(pose_classified_data, pose_test_labels)

%% k-Nearest Neighbor
k = 1;
pose_classified_data = kNearestNeighbor(k, Rpose_train_data, pose_train_labels, Rpose_test_data, 68);
accuracy = classifierAccuracy(pose_classified_data, pose_test_labels)

%% MDA
MDA_pose_train_data = MDA_function_2(pose_train_data, pose_train_labels, 'pose');
MDA_pose_test_data = MDA_function_2(pose_test_data, pose_test_labels, 'pose');

%% Bayes Classifier
pose_classified_data = bayesClassifier(MDA_pose_train_data, pose_train_labels, MDA_pose_test_data, 'pose');
accuracy = classifierAccuracy(pose_classified_data, pose_test_labels)

%% k-Nearest Neighbor
k = 1;
pose_classified_data = kNearestNeighbor(k, MDA_pose_train_data, pose_train_labels, MDA_pose_test_data, 68);
accuracy = classifierAccuracy(pose_classified_data, pose_test_labels)