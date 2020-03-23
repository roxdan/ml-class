function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% All solutions are vectorized to avoid using loops.

% Predict

predict = X * theta;

% Calculate squared error

errorsSqr = (predict-y).^2;

% Regularization, we set theta0 to 0 because it can't be regularized

theta(1) = 0;

reg = (lambda / (2 * m)) * sum(theta.^2);

% Cost function + regularization term

J = (1 / (2 * m)) * sum(errorsSqr) + reg;

% Regularized gradient cost

reg_grad = theta  * (lambda / m);

grad = ((X' * (predict-y)) / m) + reg_grad;











% =========================================================================

grad = grad(:);

end
