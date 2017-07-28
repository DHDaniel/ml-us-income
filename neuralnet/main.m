% This program implements neural networks to build a classifier that predicts whether a person in the United States makes more or less than $50,000 a year based on age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. Data is courtesy of UCI.



%% Initialization
clear ; close all; clc

% ==========================
% LOADING AND INITIALIZING PARAMETERS
% ==========================

% loading our data and naming our variables
rawData = csvread("adult.data.processed.csv");

num_samples = length(rawData);

bound1 = round(num_samples * 0.6);
bound2 = round(num_samples * 0.8);

dataTrain = rawData(1:bound1, :);
dataVal = rawData(bound1 + 1:bound2, :);
dataTest = rawData(bound2 + 1:end, :);

% setting all the variables we will use for data and normalizing
X_train = [dataTrain(:, 1:end - 1)];
[X_train, mu, sigma] = normalize(X_train);
X_train = [ones(size(dataTrain, 1), 1) X_train];
y_train = dataTrain(:, size(dataTrain, 2));


X_val = [dataVal(:, 1:end - 1)];
X_val = normalizeWith(X_val, mu, sigma);
X_val = [ones(size(dataVal, 1), 1) X_val];
y_val = dataVal(:, size(dataVal, 2));

X_test = [dataTest(:, 1:end - 1)];
X_test = normalizeWith(X_test, mu, sigma);
X_test = [ones(size(dataTest, 1), 1) X_test];
y_test = dataTest(:, size(dataTest, 2));


input_layer_size = 14;
hidden_layer_size = 30;
output_layer_size = 1;


% ================================
% OPTIMIZING THETA PARAMETERS
% ================================

% get row and column number
[r c] = size(X_train);


% checking implementation of backpropagation using numerical gradients
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
%checkNNGradients;

fprintf('Program paused. Press enter to continue.\n');
pause;

% checking backpropagation using numerical gradients and regularization.
fprintf('\nChecking Backpropagation with regularization... \n');

%  Check gradients by running checkNNGradients
%checkNNGradients(3);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nTraining Neural Network... \n')

lambda = 1;
iterations = 200;

[Theta1, Theta2] = trainNeuralNet(input_layer_size, hidden_layer_size, output_layer_size, X_train, y_train, lambda, iterations);


fprintf('Program paused. Press enter to continue.\n');
pause;


% ================================
% ACCURACY CHECKING
% ================================

y_train_pred = predict(Theta1, Theta2, X_train);
y_val_pred = predict(Theta1, Theta2, X_val);
y_test_pred = predict(Theta1, Theta2, X_test);

printf('Testing accuracy of neural network... \n\n');

printf('Training set accuracy: %f \n', mean(double(y_train_pred == y_train)) * 100);
printf('Validation set accuracy: %f \n\n', mean(double(y_val_pred == y_val)) * 100);
printf('Training set accuracy: %f \n\n', mean(double(y_test_pred == y_test)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ==================================
% LEARNING CURVE OPTIMIZATION
% ==================================

printf('Plotting learning curve...\n');
printf('Training thetas...\n');

lambda = 0;
% Computing error values
[error_train, error_val] = nnLearningCurve(X_train, y_train, X_val, y_val, input_layer_size, hidden_layer_size, output_layer_size, lambda);

% number of error samples
errsamples = size(error_train, 1);

figure(1);
plot(1:errsamples, error_train, 1:errsamples, error_val);
title('Learning curve for neural net')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
%axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:errsamples
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


printf('Plotting validation curve...\n');
printf('Training thetas...\n');

% Computing error values
[lambda_vec, error_train, error_val] = validationCurve(X_train, y_train, X_val, y_val, input_layer_size, hidden_layer_size, output_layer_size);

% number of error samples
errsamples = size(error_train, 1);

figure(2);
plot(lambda_vec, error_train, lambda_vec, error_val);
title('Validation curve for neural net')
legend('Train', 'Cross Validation')
xlabel('Lambdas')
ylabel('Error')

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
