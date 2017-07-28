function [Theta1, Theta2] = trainNeuralNet(input_size, hidden_size, output_size, X, y, lambda, iterations)
% Trains the neural network on the specific X and y examples, and performs the number of iterations specified to minimize the function. Returns the unrolled theta parameters Theta1 and Theta2
% input_size, output_size and hidden_size are the respective layer sizes of the neural network, used in the cost function and to unroll and re-roll the parameters.
% Lambda is the regularization to use.

% randomly initialize the theta parameters
Theta1 = randWeights(hidden_size, input_size);
Theta2 = randWeights(output_size, hidden_size);
initial_thetaVec = [Theta1(:) ; Theta2(:)];

% set iteration number
options = optimset('MaxIter', iterations);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) neuralCostFunction(p, ...
                                   input_size, ...
                                   hidden_size, ...
                                   output_size, X, y, lambda);


 % Now, costFunction is a function that takes in only one argument (the neural network parameters)
 [thetaVec, cost] = fmincg(costFunction, initial_thetaVec, options);

 % Obtain Theta1 and Theta2 back from nn_params
 Theta1 = reshape(thetaVec(1:hidden_size * (input_size + 1)), hidden_size, (input_size + 1));

 Theta2 = reshape(thetaVec((1 + (hidden_size * (input_size + 1))):end), output_size, (hidden_size + 1));

end
