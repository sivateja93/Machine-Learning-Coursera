function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = transpose(theta) * transpose(X);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

for i=1:m
    J = J + (0.5/m)*(h(i)-y(i))^2;
end

for j=2:n
    J = J + ((lambda/(2*m))*(theta(j)^2) );
end

for i=1:m
    for j = 1:n
        grad(j) = grad(j) + (1/m)*(h(i) - y(i))*X(i,j) ;
    end  
end

for j = 2:n    
    grad(j) = grad(j) + (lambda/m)*theta(j);
end



% =========================================================================

grad = grad(:);

end
