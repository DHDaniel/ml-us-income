function [J grad] = neuralCostFunction(paramVec, input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda)
% Computes the cost function of the neural network and the gradients. paramVec is a vector containing all the "unrolled" weights of the neural net, which we can re-obtain by using the reshape command with the input layer size, hidden layer size, and output layer size.
% NOTE THAT THE X PROVIDED ALREADY HAS THE BIAS UNIT ADDED.

m = length(y);

% obtaining the weights using the dimensions provided (this is a one hidden layer architecture)
Theta1 = reshape(paramVec(1:(hidden_layer_size * (input_layer_size + 1))), hidden_layer_size, input_layer_size + 1);

Theta2 = reshape(paramVec(1 + (hidden_layer_size * (input_layer_size + 1)):end), output_layer_size, hidden_layer_size + 1);

% ===============================
% FORWARD PROPAGATION
% ===============================

% forward propagation to get the final h(x) for each input.
% This implementation is easily vectorized because it has only one output node.

% The X provided ALREADY HAS the bias unit
a1 = X';

z2 = Theta1 * a1;

a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)) ; a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

output = a3';

% our neural network only has one output unit, so the sum over output units K isn't necessary.
J = (1 / m) * sum((-y .* log(output)) - ((1 - y) .* log(1 - output)));

% not regularizing first column of each matrix since those are the theta that correspond to the bias terms.
reg_term = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

J += reg_term;


% ============================
% BACKPROPAGATION ALGORITHM
% ============================

% implementing backprop to get the gradient of the cost function.
% implementation is vectorized easily because of the fact that there is only one output unit.

a3 = output';

d3 = a3 - y';

% omitting d2_0, corresponding to bias unit.
d2 = (Theta2' * d3)(2:end, :) .* sigmoidGradient(z2);

Theta1_accumul = d2 * a1';
Theta2_accumul = d3 * a2';

Theta1_reg = (lambda / m) .* Theta1;
Theta2_reg = (lambda / m) .* Theta2;

Theta1_grad = ((1 / m) .* Theta1_accumul) + Theta1_reg;
Theta2_grad = ((1 / m) .* Theta2_accumul) + Theta2_reg;

% not regularizing the first column of Theta matrices, as these correspond to the bias units.
Theta1_grad(:, 1) -= Theta1_reg(:, 1);
Theta2_grad(:, 1) -= Theta2_reg(:, 1);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
