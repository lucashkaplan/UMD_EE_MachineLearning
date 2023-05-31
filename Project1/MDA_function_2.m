% inputs: x = input data, input_data_labels, dataset
function MDA_data = MDA_function_2(input_data, input_data_labels, dataset)
% FACE data
if(strcmpi(dataset, 'FACE'))
    num_of_classes = 2;
end
% POSE data
if(strcmpi(dataset, 'POSE'))
    num_of_classes = 68;
end

% work with transpose of input data
x = input_data.';

% number of samples and number of features in input data
[numFeatures, numSamples] = size(x);

% within class scatter
S_W = zeros(numFeatures, numFeatures);
% between class scatter
S_B = zeros(numFeatures, numFeatures);
% scatter
S_i = zeros(numFeatures, numFeatures, num_of_classes);
% class means
classMean = zeros(numFeatures, num_of_classes);

% number of samples in each class
classSampleNum = zeros(num_of_classes, 1);

% total sum of samples
sumSampleVal = zeros(numFeatures, 1);

% loop once per class
for w_i = 1:num_of_classes
    % loop through all input data
    for data_index = 1:numSamples
        %  if img. is in class w_i, place it in class_data 
        if (input_data_labels(data_index) + 1) == w_i
            class_data(:, classSampleNum(w_i) + 1) = x(:, data_index);
            % sum of all samples
            sumSampleVal = sumSampleVal + class_data(:, classSampleNum(w_i) + 1);
            % increase number of samples in class
            classSampleNum(w_i) = classSampleNum(w_i) + 1;
        end 
    end 
    
    %{
    %%% calculate scatter for class w_i = cov(X_i)
    S_i(:, :, w_i) = cov(class_data.') + ((10^-8)*eye(numFeatures));

    % find mean for class w_i
    classMean(:, w_i) = mean(class_data, 2); % returns column vector w/ mean of each dimension
    %}

    
    % find mean for class w_i
    classMean(:, w_i) = mean(class_data, 2); % returns column vector w/ mean of each dimension 
    
    % loop through all samples in class w_i
    for k = 1:classSampleNum(w_i)
        % calc. scatter for class w_i
        classMeanDiff = class_data(:, k) - classMean(:, w_i);
        S_i(:, :, w_i) = S_i(:, :, w_i) + (classMeanDiff * (classMeanDiff.')); 
    end
    % add value on diagonal so matrix non-singular
    S_i(:, :, w_i) = S_i(:, :, w_i) + ((10^-7)*eye(numFeatures));
end

%%% w/in class scatter
% S_W = S_1 + S_2 + ... + S_C
S_W = sum(S_i, 3);

%%% b/w class scatter
% total mean vec.
m_T = (1/numSamples)*sumSampleVal;

% S_B = sum from i to C of (n_i*(m_i - m)*(m_i - m))
% m = overall mean, m_i = mean of class i, n_i = number of samples in class i
for i = 1:num_of_classes
    mean_diff = classMean(:, i) - m_T;
    S_B = S_B + (classSampleNum(w_i) * (mean_diff * (mean_diff.')));
end

%%% find m largest eigenvectors corresponding to m largest eigenvalues
% max data separability found by m <= C - 1
m = num_of_classes - 1;

A = inv(S_W)*S_B;
% find eigenvectors and eigenvalues of S_W^-1*S_B
[eigenvectors, eigenvalues] = eig(A);
% make vector of eigenvalues (eigenvalues stored on diagonal)
eigenvalue_vec = diag(eigenvalues);
% find max m eigenvalues and their indices
[~, max_eigenvalue_index] = maxk(eigenvalue_vec, m);

% create matrix of eigenvectors corresponding to max m eigenvalues
eigenvector_matrix = eigenvectors(:, max_eigenvalue_index);

% transform data
MDA_data = real(((eigenvector_matrix.')*x).');

end