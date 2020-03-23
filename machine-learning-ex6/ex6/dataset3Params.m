function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

# Create a list to test all values for C and Sigma

c_aux = C;
sigma_aux = sigma;

list_aux = [0.01 0.03 0.1 0.3 1 3 10 30];

# Results

results = zeros(length(list_aux) * length(list_aux), 3);

# For loop to test all C's and Sigma's to find the minimum error value.

i = 1;
for c_aux = list_aux
  for sigma_aux = list_aux
    model = svmTrain(X, y, c_aux, @(x1, x2) gaussianKernel(x1, x2, sigma_aux));
    predictions = svmPredict(model, Xval);
    error_aux = mean(double(predictions ~= yval));
    results(i,:) = [c_aux sigma_aux error_aux];
    i = i + 1;
  endfor
endfor

# Find the minimum and its indexes

[v i] = min(results(:,3));

C = results(i,1);
sigma = results(i,2);

fprintf('Optimal C : %f\n', C);
fprintf('Optimal Sigma: %f\n', sigma);
fprintf('Optimal Error: %f\n', results(i,3));

% =========================================================================

end
