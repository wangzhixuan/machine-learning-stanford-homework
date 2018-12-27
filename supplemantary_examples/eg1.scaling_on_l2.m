clear ; close all; clc

%% ==================== Part 1 ====================

% load data
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 2 ====================
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

num_iters = 50;

for lambda = [0 1 10],
  for x2_scaling = [1 0.01],
    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);
    
    % scale 
    X_new = X;
    X_new(:, 2) = X(:, 2)*x2_scaling;

    % Optimize
    [theta, J, exit_flag] = ...
      fminunc(@(t)(costFunctionReg(t, X_new, y, lambda)), ...
              initial_theta, options);
  
    % Optimize
    %theta = initial_theta;
    %for iter = 1:num_iters
    %
    %    alpha = 10.0/iter;
    %
    %    y_pred = X * theta;
    %    [J, gradient] = costFunctionReg(theta, X_new, y, lambda);
    %    theta = theta - alpha * gradient;
    %    fprintf('===============\n');
    %    fprintf('theta = %g %g %g\n', theta);
    %    fprintf('gradient = %g %g %g\n', gradient);
    %    fprintf('J = %g\n', J);
        
    %end     
    
    
    % scale coeffient back
    theta_scaled_back = theta;
    theta_scaled_back(2) /= x2_scaling;
    
    plotDecisionBoundary(theta, X_new, y);
    hold on;
    title(sprintf('lambda = %g, theta = %g %g %g\n', lambda, theta))
    fprintf('\nlambda = %g, theta = %g %g %g, J=%g\n', lambda, theta, J)

  endfor
endfor
