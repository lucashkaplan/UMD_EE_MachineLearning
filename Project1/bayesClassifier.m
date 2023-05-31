function classified_data = bayesClassifier(training_data, training_data_labels, testing_data, dataset)

% FACE data
if(strcmpi(dataset, 'FACE'))
    num_of_classes = 2;
end
% POSE data
if(strcmpi(dataset, 'POSE'))
    num_of_classes = 68;
end

% number of samples in training data
n_total = size(training_data, 1);

% normalizing training and testing data s.t. the mean for each img (row) = 0
% and its std. deviation is 1
training_data = zscore(training_data, 0, 2);
testing_data = zscore(testing_data, 0, 2);

inputDataDimension = size(training_data, 2);

organized_training_data = zeros(size(training_data, 1), size(training_data, 2));
mu = zeros(num_of_classes, size(training_data, 2));
classified_data = zeros(1, size(testing_data, 1));

% number of samples in each class
classSamples = zeros(num_of_classes, 1);
% index for placing data in organized matrix
organized_index = 1;
% run through training data once per class
for w_i = 1:num_of_classes
    for data_index = 1:size(training_data, 1)
        %  if img. is in class w_i, place it in organized_training_data 
        if (training_data_labels(data_index) + 1) == w_i
            organized_training_data(organized_index, :) = training_data(data_index, :);
            % increase index counter and number of samples in class
            organized_index = organized_index + 1;
            classSamples(w_i) = classSamples(w_i) + 1;
        end 
    end
end

% find mean and covariance for each class
% intialize covariance matrix
sigma = zeros(inputDataDimension, inputDataDimension, num_of_classes);
% class 1
if(classSamples(1) ~= 0)
    mu(1, :) = mean(organized_training_data(1:classSamples(1), :));
    sigma(:, :, 1) = cov( organized_training_data(1:classSamples(1), :) ) + ((10^-8)*eye(inputDataDimension));
else
    sigma(:, :, 1) = ((10^-8)*eye(inputDataDimension));
end
% rest of classes
for class = 2:num_of_classes
    % check if have any samples in class
    if(classSamples(class) ~= 0)
        mu(class, :) = mean(organized_training_data(((classSamples(class - 1) + 1):(classSamples(class - 1) + classSamples(class))), :));
        sigma(:, :, class) = cov( organized_training_data(((classSamples(class - 1) + 1):(classSamples(class - 1) + classSamples(class))), :) ) + ((10^-8)*eye(inputDataDimension));
    end
end

P = classSamples/n_total; % prior probabilities

% intialize all discriminants to be NaN, so that they are treated as
% neg. infinity
for i = 1:num_of_classes
    g(i) = 0/0;
end
% classify each img. (row) into a class
for data_index = 1:size(testing_data, 1)
    % classifier for each class that has at least 1 sample
    for i = 1:num_of_classes
        if(classSamples(i) > 0)
            g(i) = (-0.5 * log(det(sigma(:, :, i)))) - (0.5 * (testing_data(data_index, :) - mu(i, :)) * (inv(sigma(:, :, i))) * ((testing_data(data_index, :) - mu(i, :)).') ) + log(P(i));
        end
    end

    % assign img. (row) to largest classifier
    [~, classifierNum] = max(g);
    classified_data(1, data_index) = classifierNum - 1; 
end

end