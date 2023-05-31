%% ENEE436 Project 1 - Task 2
% Author: Lucas Kaplan

% Goal: make a classifier that given an unseen facial image, tells us 
% whether it is a neutral face or has a facial expression

% FACE training data:
% [img1: [pixel_1 ... pixel_504]]
% [img2: [pixel_1 ... pixel_504]] 
% ...
% [img319: [pixel_1 ... pixel_504]]
% [img320: [pixel_1 ... pixel_504]]

%% Load FACE Data  
% Dataset
data_folder = './Data/';

% Test Ratio
test_ratio = 0.2;

load([data_folder,'data.mat']) % load in 3D face array
Ns = 200; % number of images
% dataset: neutral face | facial expression | illumination variations
face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);
face_il = face(:,:,3:3:3*Ns);

i = randi([1,Ns],1);

% Convert the dataset in data vectors and labels for
% netutral vs facil expression classification

data = [];
labels = [];
[m,n] = size(face_n(:,:,i));
for subject=1:Ns
    % neutral face: label 0
    face_n_vector = reshape(face_n(:,:,subject),1,m*n);
    data = [data ; face_n_vector];
    labels = [labels 0];
    % face with expression: label 1
    face_x_vector = reshape(face_x(:,:,subject),1,m*n);
    data = [data ; face_x_vector];
    labels = [labels 1];  
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
face_train_data = data(idx(1:N),:);
face_train_labels = labels(idx(1:N));
face_test_data = data(idx(N+1:2*Ns),:);
face_test_labels = labels(idx(N+1:2*Ns));

%% PCA
% desired number of reduced dimensions
reducedDim = 85;

eigenvec_matrix = pca(face_train_data);
Rface_train_data = face_train_data * (eigenvec_matrix(:, 1:reducedDim));
Rface_test_data = face_test_data * (eigenvec_matrix(:, 1:reducedDim));

%% Bayes Classifier
face_classified_data = bayesClassifier(Rface_train_data, face_train_labels, Rface_test_data, 'FACE');
accuracy = classifierAccuracy(face_classified_data, face_test_labels)

%% k-Nearest Neighbor
k = 1;
face_classified_data = kNearestNeighbor(k, Rface_train_data, face_train_labels, Rface_test_data, 2);
accuracy = classifierAccuracy(face_classified_data, face_test_labels)

%% MDA
MDA_face_train_data = MDA_function_2(face_train_data, face_train_labels, 'FACE');
MDA_face_test_data = MDA_function_2(face_test_data, face_test_labels, 'FACE');

%% Bayes Classifier
face_classified_data = bayesClassifier(MDA_face_train_data, face_train_labels, MDA_face_test_data, 'FACE');
accuracy = classifierAccuracy(face_classified_data, face_test_labels)

%% k-Nearest Neighbor
k = 1;
face_classified_data = kNearestNeighbor(k, MDA_face_train_data, face_train_labels, MDA_face_test_data, 2);
accuracy = classifierAccuracy(face_classified_data, face_test_labels)


