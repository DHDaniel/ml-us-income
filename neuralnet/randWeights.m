function [Theta] = randWeights(L_out, L_in)
% randomly initializes the weights of a neural network based on the number of input and output units of the corresponding layers.

epsilon_init = sqrt(6) / (sqrt(L_in) + sqrt(L_out));

% generates a random matrix in the range [-epsilon_init, epsilon_init]
Theta = rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init;

end
