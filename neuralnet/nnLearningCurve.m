function [error_train, error_val] = nnLearningCurve(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, lambda)
% Returns two vectors of the same length, indicating the error in the training set and the validation set. This helps diagnose errors like high bias or high variance, or a combination of both.

m = size(X_train, 1);

error_train = [];
error_val = [];

for i = 1:1250:m

  Xsub = X_train(1:i, :);
  ysub = y_train(1:i, :);

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
