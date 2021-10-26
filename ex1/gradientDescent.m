function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %h = X * theta;
    %t = h - y;
    %theta  = theta -(alpha/m) * X' * t;
    % h = (X * theta);
    % mid_div = h - y;
    % theta = theta - ((alpha * (X'*mid_div))/m);
    

    % linear/scalar solution
    % h = X * theta;
    % for i=1:size(X)(1),
    %   for j=1:size(X)(2),
    %     Y = y(i, 1);
    %     x = X(i, j);
    %     H = h(i, 1);
    %     theta(j,1) = theta(j,1) - ( (alpha/m) * (x * (H - Y) ) );
    %   endfor;  
    % endfor;
    
    % parametized solution
    h = X * theta;
    theta = theta - ((alpha/m) * (X'*(h-y))); 


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
