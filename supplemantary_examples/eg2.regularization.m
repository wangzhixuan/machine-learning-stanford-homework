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

% scale X2 so that it is not that symmetric
X(:,2) = X(:,2) * 0.333;
X(:,3) = X(:,3) * 0.1;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 2 ====================
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

initial_theta = zeros(size(X, 2), 1);
      
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============

fprintf('Visualizing J(theta_1, theta_2) ...\n')

% Grid over which we will calculate J
theta1_vals = linspace(-0.2, 1, 100);
theta2_vals = linspace(-0.2, 2, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta1_vals), length(theta2_vals));

theta_series = zeros(2,4,2);
i0=1;

[theta, J, exit_flag0] = ...
fminunc(@(t)(costFunctionReg(t, X, y, 0, 2)), ...
        initial_theta, options);
    
theta_series(:, 1, 1) = theta(2);
theta_series(:, 1, 2) = theta(3);

indecies = [0.5 4];
for index=indecies,
  j0=2;
  for lambda = [1 3 10],
    fprintf('index=%g lambda=%g\n', index, lambda);
    [theta, J, exit_flag0] = ...
    fminunc(@(t)(costFunctionReg(t, X, y, lambda, index)), ...
            initial_theta, options)
    % Fill out J_vals
    for i = 1:length(theta1_vals)
        for j = 1:length(theta2_vals)
          [theta0, J_vals(i,j), exit_flag] = ...
              fminunc(@(t)(costFunction2DReg(t, theta1_vals(i), theta2_vals(j), X, y, lambda, index)), 0, options);
          %t = [theta0; theta1_vals(i); theta2_vals(j)];
          %J_vals(i,j) = costFunctionReg(t, X, y, lambda);
        end
        fprintf('=========\n')
    end
    
    if (1)#(exit_flag0 !=3)
      [aa,bb] = min(J_vals(:));
      [ii,jj] = ind2sub(size(J_vals), bb);
      theta(2) = theta1_vals(ii);
      theta(3) = theta2_vals(jj);
    endif
    
    % Because of the way meshgrids work in the surf command, we need to
    % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals';
    % Surface plot
    if (index < 1) & (lambda > 0)
      figure;
      surf(theta1_vals, theta2_vals, J_vals)
      xlabel('\theta_1'); ylabel('\theta_2');
    endif
    
    % Contour plot
    figure;
    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta1_vals, theta2_vals, J_vals, logspace(-2, 3, 50), 'ShowText', 'On')
    xlabel('\theta_1'); ylabel('\theta_2');
    hold on;
    plot([theta(2)], [theta(3)], 'r+', 'MarkerSize', 10)
    plot([-1,10], [-1,10], 'r--')
    plot([-1,10], [0,0], 'r--')
    title(...
        sprintf(...
           'power index=%g, lambda=%g, theta(2:3)=%g %g\n', ...
           index, lambda, theta(2), theta(3) ...
        ) ...
    )
    %title(sprintf('lambda = %g, theta = %g %g %g\n', lambda, theta));
    
    theta_series(i0,j0, :) = theta(2:3)
    j0+= 1;
    
  endfor
  i0+= 1;
endfor

theta_series
%theta_series = zeros(2,4,2)

%theta_series(:,:,1) = [
%   [2.06230   1.84444   1.51111   0.80000];
%   [2.06230   1.53333   1.26667   1.00000]
%]
%theta_series(:,:,2) =[
%   [0.60503   0.53939   0.44242   0.24848];
%   [0.60503   0.46667   0.39394   0.33333]
%]

%options = optimset('GradObj', 'on', 'MaxIter', 400);
%

% Grid over which we will calculate J
theta1_vals = linspace(-0.2, 1, 30);
theta2_vals = linspace(-0.2, 2.5, 30);

index=2
lambda=0

J_vals = zeros(length(theta1_vals), length(theta2_vals));
for i = 1:length(theta1_vals)
    for j = 1:length(theta2_vals)
      [theta0, J_vals(i,j), exit_flag] = ...
          fminunc(@(t)(costFunction2DReg(t, theta1_vals(i), theta2_vals(j), X, y, lambda, index)), 0, options);
      %t = [theta0; theta1_vals(i); theta2_vals(j)];
      %J_vals(i,j) = costFunctionReg(t, X, y, lambda);
    end
    fprintf('=========\n')
end

J_vals = J_vals';
figure;

contour(theta1_vals, theta2_vals, J_vals, logspace(-2, 3, 50), 'ShowText', 'On')
xlabel('\theta_1'); ylabel('\theta_2');
title('the different trajectory of theta from different exponent(0.5 vs 4) when lambda increases')
hold on;
for i=1:2,
  plot(theta_series(i, :, 1), theta_series(i, :, 2), '+--', 'MarkerSize', 10);
endfor
plot([-1, 10], [-1, 10], 'k-.')
plot([0, 0], [-1, 10], 'k-.')

legend('contour', 'index=0.5', 'index=4');
