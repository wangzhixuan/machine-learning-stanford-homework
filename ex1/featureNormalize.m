function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X, dim=1);
sigma = std(X, opt=0, dim=1);

m = size(X, 1);

mu_repeated = repmat(mu, m, 1)
sigma_repeated = repmat(sigma, m, 1)

#X-mu_repeated

% Print out some data points
fprintf(' mu = %f \n', mu)
fprintf(' sigma = %f \n', sigma)

#fprintf('First 10 examples from the dataset: \n');
#fprintf(' x = [%.0f %.0f] \n', [(X-mu_repeated)(1:10,1:2)]);

X_norm = (X-mu_repeated)./sigma_repeated

% ============================================================




end
