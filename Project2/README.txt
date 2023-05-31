Goal: Classify handwritten digits using multi-class linear SVM.
Method: The images containing each handwritten digit are taken from the MNIST dataset. This data is loaded, after which
the linear SVM can be trained and used to classify the test data. If desired, one can use the "PCA" section to reduce the
dimensions of the MNIST data, which reduces the classifier's runtime, but also reduces its accuracy.

Note: Before running the classifier (ENEE436_Proj2.m), the SVM library must be unzipped, and this folder must be placed
in the MATLAB working directory. The MNIST dataset is available online at: https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/