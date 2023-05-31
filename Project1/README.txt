Task 1 (ENEE436_Proj1_Task1.m)
Goal: Classify the subject of a test image.
Method: For this task, the POSE data set is used, which contains 68 subjects under 13 different poses. The data was split
such that a random 80% of the images were used for training, and 20% were used for testing. Within this task, there were 2
subtasks. 
Subtask 1: Apply PCA to the input data and then classify the test data using the k-Nearest Neighbor approach and
the Bayes' Classification. The parameter reducedDim can be varied to change the final number of dimensions in the data.
Subtask 2: Apply MDA to the input data and then classify the test data using the k-Nearest Neighbor 
approach and the Bayes' Classification. To vary the number of dimensions reduced by MDA, the variable m within MDA_function_2 
must be changed.

Task 2 (ENEE436_Proj1_Task2.m)
Goal: Create a classifier that given an test facial image, tells us whether it is a neutral face or has a facial expression.
Method: For this task, the FACE data set is used, which contains 200 subjects and 3 images per subject. For each subject, an 
image of a neutral face, facial expression, and illumination variation of the face are provided.
Subtask 1: Apply PCA to the input data and then classify the test data using the k-Nearest Neighbor approach and
the Bayes' Classification. The parameter reducedDim can be varied to change the final number of dimensions in the data.
Subtask 2: Apply MDA to the input data and then classify the test data using the k-Nearest Neighbor 
approach and the Bayes' Classification. To vary the number of dimensions reduced by MDA, the variable m within MDA_function_2 
must be changed.