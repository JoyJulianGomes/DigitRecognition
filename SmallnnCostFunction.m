function [J, grad] = SmallnnCostFunction(nn_params, ...
                                   Input_Layer_Size, ...
                                   Hidden_Layer_1_Size, ...
                                   Output_Layer_Size, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:Hidden_Layer_1_Size * (Input_Layer_Size + 1)), ...
                 Hidden_Layer_1_Size, (Input_Layer_Size + 1));

Theta2 = reshape(nn_params((1 + (Hidden_Layer_1_Size * (Input_Layer_Size + 1))):end), ...
                 Output_Layer_Size, (Hidden_Layer_1_Size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
%J = 0;
%Theta1_grad = zeros(size(Theta1));%25:401
%Theta2_grad = zeros(size(Theta2));%10:26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

%Converting y from single column label to 10 dimenional label


%Vectorized Implementation
yVec = (0:Output_Layer_Size-1) == y; %Creates a matrix containing values 1,2..,num_labels
                            %created matrix is comapared with y value;
                            %comparison returns logical 1 or 0
                            %Idea taken from Resource section of Coursera
                            %Machine learning course.

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

A1 = [ones(m, 1), X];

z2 = A1 * Theta1';
g_z2 = sigmoid(z2);
A2 = [ones(m,1), g_z2];

z3 = A2 * Theta2';
g_z3 = sigmoid(z3);
A3 = g_z3; % = h_theta(X) its now a 5000x10 matrix of prediction/hypothesis

SquaredTheta1 = sum(sum(Theta1( : , 2:end).^2));
SquaredTheta2 = sum(sum(Theta2( : , 2:end).^2));
J = sum(sum((-yVec .* log(A3))-((1-yVec) .* log(1 - A3))))/m ...
    + (lambda*(SquaredTheta1+SquaredTheta2))/(2*m);


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

    d3 = A3 - yVec; %d3:5000x10 A3:5000x10 yVec:5000x10
    %d2 = (d3 * Theta2) .* sigmoidGradient(A1 * Theta1');
    %d2:5000x26 <- Wrong the bias term will always provide 1 value
    %d2:5000x25 d3:5000x10 Theta2:10x26 A1:5000x401 Theta1':401x25 
    %           d3*Theta2:5000x26 A1*Theta1': 5000x25
    %                            ^-Dimensional conflict need to ignore 1st
    %                            row of d3*Theta2 which holds value for
    %                            bias activation
    d3T2 = (d3 * Theta2);
    d2 = d3T2(:,2:end) .* sigmoidGradient(z2);
    %fprintf("SIize of d2 at line nnCostFunction.102 = \n");
    %fprintf("%f", size(d2));

    D2 = (d3' * A2)/m;
    %D2:10x26 d3':10x5000 A2:5000x26
    D1 = (d2' * A1)/m;
    %D1:25x401 d2':25x5000 A1:5000x401
    
    Theta1_grad = D1;
    Theta2_grad = D2;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
RegTheta1 = (lambda/m)*Theta1;
RegTheta1(:, 1) = 0;
RegTheta2 = (lambda/m)*Theta2;
RegTheta2(:, 1) = 0;

Theta1_grad = Theta1_grad + RegTheta1;
Theta2_grad = Theta2_grad + RegTheta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
