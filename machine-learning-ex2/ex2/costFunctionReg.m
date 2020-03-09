function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hyphotesis = sigmoid(X * theta);
  
%disp("That's the hyphotesis: ");
%disp(hyphotesis);
%logt = zeros(size(X,2));

% Calculating the cost Function (regularizated and vectorized)

% Create a theta without theta0 (we don´t apply regularization on him)
theta_x = theta(2:end,:);

% Auxiliar variate to store the regularization coeficient
aux = (lambda / (2 * m)) * (theta_x' * theta_x);

J = (-1 ./ m) * (y' * log(hyphotesis) + (1 - y)' * log(1 - hyphotesis)) + aux;

% Calculating the gradient (vetorized)

aux_2 = ones(size(theta));
aux_2(1) = 0;

grad = (X' * (hyphotesis - y)) ./ m + lambda * (theta .* aux_2) / m;

% =============================================================

end
