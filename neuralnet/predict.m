function [y] = predict(Theta1, Theta2, X)
% Function uses forward propagation to predict the class of inputs X. Note that X already has the bias unit added to it. "y" is a column vector with the labels "1" or "0", corresponding to the training example in the ith row of X.


a1 = X';

z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)) ; a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

y = a3';

y = round(y);

end
