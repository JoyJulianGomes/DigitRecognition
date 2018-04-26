function [J grad] = LargennCostFunction(nn_params, ...
                                        Input_Layer_Size, ...
                                        Hidden_Layer_1_Size, ...
                                        Hidden_Layer_2_Size, ...
                                        Hidden_Layer_3_Size, ...
                                        Output_Layer_Size, ...
                                        X, y, lambda)
[theta_1, theta_2, theta_3, theta_4] = LargeRollParams(nn_params, ...
                                                       Input_Layer_Size,...
                                                       Hidden_Layer_1_Size,...
                                                       Hidden_Layer_2_Size,...
                                                       Hidden_Layer_3_Size,...
                                                       Output_Layer_Size);
% Setup some useful variables
m = size(X, 1);
yVec = (0:Output_Layer_Size-1) == y;

A1 = [ones(m, 1), X];

z2 = A1 * theta_1';
g_z2 = sigmoid(z2);
A2 = [ones(m,1), g_z2];

z3 = A2 * theta_2';
g_z3 = sigmoid(z3);
A3 = [ones(m,1), g_z3];

z4 = A3 * theta_3';
g_z4 = sigmoid(z4);
A4 = [ones(m,1), g_z4];

z5 = A4 * theta_4';
g_z5 = sigmoid(z5);
A5 = g_z5; % = h_theta(X) its now a 5000x10 matrix of prediction/hypothesis

SquaredTheta1 = sum(sum(theta_1( : , 2:end).^2));
SquaredTheta2 = sum(sum(theta_2( : , 2:end).^2));
SquaredTheta3 = sum(sum(theta_3( : , 2:end).^2));
SquaredTheta4 = sum(sum(theta_4( : , 2:end).^2));

J = sum(sum((-yVec .* log(A5))-((1-yVec) .* log(1 - A5))))/m ...
    + (lambda*(SquaredTheta1+SquaredTheta2+SquaredTheta3+SquaredTheta4))/(2*m);


d5 = A5 - yVec;
d5t4 = d5 * theta_4;

d4 = d5t4(:,2:end) .* sigmoidGradient(z4);
d4t3 = d4 * theta_3;

d3 = d4t3(:,2:end) .* sigmoidGradient(z3);
d3T2 = (d3 * theta_2);

d2 = d3T2(:,2:end) .* sigmoidGradient(z2);
    
D4 = (d5' * A4)/m;
D3 = (d4' * A3)/m;
D2 = (d3' * A2)/m;
D1 = (d2' * A1)/m;

Theta1_grad = D1;
Theta2_grad = D2;
Theta3_grad = D3;
Theta4_grad = D4;

RegTheta1 = (lambda/m)*theta_1;
RegTheta1(:, 1) = 0;
RegTheta2 = (lambda/m)*theta_2;
RegTheta2(:, 1) = 0;
RegTheta3 = (lambda/m)*theta_3;
RegTheta3(:, 1) = 0;
RegTheta4 = (lambda/m)*theta_4;
RegTheta4(:, 1) = 0;

Theta1_grad = Theta1_grad + RegTheta1;
Theta2_grad = Theta2_grad + RegTheta2;
Theta3_grad = Theta3_grad + RegTheta3;
Theta4_grad = Theta4_grad + RegTheta4;

grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];
end