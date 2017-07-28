function [lambda_vec, error_train, error_val] = validationCurve(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size)
% Computes the training error and validation error for different values of lambda.

% lambdas to test
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = [];
error_val = [];

for i = 1:length(lambda_vec)

  lambda = lambda_vec(i);

  % using a smaller subset of the training set for speed.
  Xsub = X_train(1:500, :);
  ysub = y_train(1:500, :);

  [Theta1, Theta2] = trainNeuralNet(input_size, hidden_size, output_size, Xsub, ysub, lambda, 50);

  % unrolling for cost function
  paramVec = [Theta1(:) ; Theta2(:)];

  % calculating the training and validation cost, respectively. We do this by using a lambda of zero (because we do not want regularization)
  [trainCost, trainGrad] = neuralCostFunction(paramVec, input_size, hidden_size, output_size, Xsub, ysub, 0);
  [valCost, valGrad] = neuralCostFunction(paramVec, input_size, hidden_size, output_size, X_val, y_val, 0);

  % adding costs to error vectors.
  error_train = [error_train ; trainCost];
  error_val = [error_val ; valCost];
endfor


end
