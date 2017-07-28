function [J grad] = costFunction(X, y, theta, lambda)
  % Calculates the logistic regression cost function given the arguments, and also returns the gradient of it. `grad` is a vector of each partial gradient with respect to theta.

  % number of training examples
  m = length(y);
  % hypothesis
  hyp = sigmoid(X * theta);

  J = (1 / m) * sum((-y .* log(hyp)) - ((1 - y) .* (log(1 - hyp))));
  reg = (lambda / (2 * m)) * sum(theta .^ 2);
  J += reg;

  grad = (1 / m) .* (X' * (hyp - y));
  grad_reg = theta .* (lambda / m);
  grad += grad_reg;
  grad(1) -= grad_reg(1);

end
