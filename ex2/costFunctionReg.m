function [J, grad] = costFunctionReg(theta, X, y, lambda, index=2)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[m,n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

y_predict = X * theta;

h = ones(size(y)) ./ (1+ exp(-y_predict));
h_bar = exp(-y_predict) ./ (1+exp(-y_predict));

[L, grad_L] = regularizor(theta(2:n), lambda/m, index);

%lambda
J = - (y'*log(h) + (1-y)'*log(h_bar))/m;
J += L;
grad = (X' * (h - y))/m;
grad(2:end) += grad_L;

%fprintf('=======labmda = %g=========\n', lambda)
%fprintf('J= %g, grad_theta=\n',J)
%fprintf('%g ', grad) 
%fprintf('\n')


% =============================================================

end
