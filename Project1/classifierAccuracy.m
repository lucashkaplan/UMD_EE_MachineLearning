function accuracy = classifierAccuracy(classified_data, test_data_labels)
numCorrect = 0;

for i = 1:length(test_data_labels)
    if classified_data(i) == test_data_labels(i)
        numCorrect = numCorrect + 1;
    end
end

accuracy = (numCorrect/(length(test_data_labels))) * 100;

end