function [X_norm] = normalizeWith(X, mu, sigma)
% Normalizes X with the provided mu and sigma values. This function should be used after normalizing the training set, to normalize the cross validation and test set using the same values.

X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
