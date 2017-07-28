% This program implements logistic regression to build a classifier that predicts whether a person in the United States makes more or less than $50,000 a year based on age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. Data is courtesy of UCI.



%% Initialization
clear ; close all; clc


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



% ================================
% OPTIMIZING THETA PARAMETERS
% ================================

% get row and column number
[r c] = size(X_train);

% remember 1s column has already been added
thetas = zeros(c, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost
[theta, cost] = fminunc(@(t)(costFunction(X_train, y_train, t)), thetas, options);

printf("Optimal thetas: \n");
printf("%f \n", theta);

printf("\n\n\n");


% ============================
% CALCULATING ACCURACY
% ============================

p_train = predict(theta, X_train);
acc_train = mean(double(p_train == y_train)) * 100;
printf("Accuracy for training set: %f \n", acc_train);

p_val = predict(theta, X_val);
acc_val = mean(double(p_val == y_val)) * 100;
printf("Accuracy for validation set: %f \n", acc_val);
