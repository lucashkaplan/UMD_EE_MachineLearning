function classified_data = kNearestNeighbor(k, training_data, training_data_labels, testing_data, num_of_classes)

% dimensions of training and testing data
[trainingSampleNum, ~] = size(training_data);
testingSampleNum = size(testing_data, 1);
% initializing output
classified_data = zeros(1, testingSampleNum);

% matrix with distance from each test point to each training pt.
% format: [dist. from test sample 1 to training sample 1, 2, 3, ... , trainingSampleNum]
%         [dist. from test sample 2 to training sample 1, 2, 3, ... , trainingSampleNum]
%         ...
%         [dist. from test sample testingSampleNum to training sample 1, 2, 3, ... , trainingSampleNum]
distMatrix = zeros(testingSampleNum, trainingSampleNum);

for i = 1:testingSampleNum
    for j = 1:trainingSampleNum
        % calculate dist. b/w training and test sample
        distMatrix(i, j) = norm(testing_data(i,:) - training_data(j, :));
    end
end

% sort rows of distMatrix to find the k smallest distances from test sample i to training samples 1, 2, ..., k
% trainingIndex stores index of sample in original row of distMatrix
[~, closestTrainingPts] = sort(distMatrix, 2, "ascend");

% iterate through all testing data
for i = 1:testingSampleNum
    % counter for number of close points in class
    classCnt = zeros(num_of_classes, 1);
    % sorted counter for number of close pts in class
    classCntSorted = zeros(num_of_classes, 1);
    % index of closest classes
    closestClassIndex = zeros(num_of_classes, 1);
    
    for j = 1:k
        % class of jth neighbor
        neighborIndex = closestTrainingPts(i, j);
        class = training_data_labels(neighborIndex) + 1;
        
        % increase counter for # of closest points for class
        classCnt(class) = classCnt(class) + 1;
    end
    
    %{
    % find class w/ most close neighbors
    [~, closestClass] = max(classCnt);
    %}
    
    %%% if there's tie, choose class randomly
    % sort classCnt vec s.t. classes w/ most close samples are at top and
    % those w/ least at bottom
    [classCntSorted, closestClassIndex] = sort(classCnt, "descend");

    % if 2 or more classes have same num of close pts
    if classCntSorted(1) == classCntSorted(2)
        % find all classes w/ same number of close pts
        equalClasses = find(classCnt == classCntSorted(1));

        % choose a random class from those with equal number of close points
        closestClass = equalClasses(randi(length(equalClasses))); 
    else
        % closest class is simply first index
        closestClass = closestClassIndex(1);
    end

    % assign test data to class with k-closest neighbors
    classified_data(1, i) = closestClass - 1; 
    
end

end