clear ; close all; clc%% ==================== Part 1 ====================% load datadata = load('ex2data1.txt');X = data(:, [1, 2]); y = data(:, 3);plotData(X, y);% Put some labels hold on;% Labels and Legendxlabel('Exam 1 score')ylabel('Exam 2 score')% Specified in plot orderlegend('Admitted', 'Not admitted')hold off;#{%  Setup the data matrix appropriately, and add ones for the intercept term[m, n] = size(X);% Add intercept term to x and X_testX = [ones(m, 1) X];fprintf('\nProgram paused. Press enter to continue.\n');pause;%% ==================== Part 2 ====================%  Set options for fminuncoptions = optimset('GradObj', 'on', 'MaxIter', 400);%% ==================== not-regularized ==================lambda = 0;x2_scaling = 1;  % Initialize fitting parameters  initial_theta = zeros(size(X, 2), 1);    % scale   X_new = X;  X_new(:, 2) = X(:, 2)*x2_scaling;      % Optimize  [theta, J, exit_flag] = ...    fminunc(@(t)(costFunctionReg(t, X_new, y, lambda)), initial_theta, options);      % scale coeffient back  theta_scaled_back = theta;  theta_scaled_back(2) /= x2_scaling;  plotDecisionBoundary(theta_scaled_back, X, y);  hold on;  title(sprintf('lambda = %g, theta = %g %g %g', lambda, theta))  fprintf('\nlambda = %g, theta = %g %g %g\n', lambda, theta)%% ==================== regularized ==================lambda = 100;% Initialize fitting parametersinitial_theta = zeros(size(X, 2), 1);% Optimize[theta, J, exit_flag] = ...	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);plotDecisionBoundary(theta, X, y);hold on;title(sprintf('lambda = %g, theta = %g %g %g', lambda, theta))fprintf('\nlambda = %g, theta = %g %g %g\n', lambda, theta)%% ==================== regularized and scaled ==================lambda = 100;% Initialize fitting parametersinitial_theta = zeros(size(X, 2), 1);% Optimize[theta, J, exit_flag] = ...	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);plotDecisionBoundary(theta, X, y);hold on;title(sprintf('lambda = %g, theta = %g %g %g', lambda, theta))fprintf('\nlambda = %g, theta = %g %g %g\n', lambda, theta)}#