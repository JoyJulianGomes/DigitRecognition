function [J, grad] = MediumnnCostFunction(nn_params, ...
                                   Input_Layer_Size, ...
                                   Hidden_Layer_1_Size, ...
                                   Hidden_Layer_2_Size, ...
                                   Output_Layer_Size, ...
                                   X, y, lambda)

m2t1r = Hidden_Layer_1_Size; m2t1c = Input_Layer_Size+1;
m2t2r = Hidden_Layer_2_Size; m2t2c = Hidden_Layer_1_Size+1;
m2t3r = Output_Layer_Size; m2t3c = Hidden_Layer_2_Size+1;
Theta1 = reshape(nn_params(1:m2t1r * m2t1c), m2t1r, m2t1c);
Theta2 = reshape(nn_params((1 + m2t1r * m2t1c):(m2t1r * m2t1c)+(m2t2r * m2t2c)), m2t2r, m2t2c);
Theta3 = reshape(nn_params((1 + (m2t1r * m2t1c)+(m2t2r * m2t2c)):end), m2t3r, m2t3c);

% Setup some useful variables
m = size(X, 1);
yVec = (0:Output_Layer_Size-1) == y;

A1 = [ones(m, 1), X];

z2 = A1 * Theta1';
g_z2 = sigmoid(z2);
A2 = [ones(m,1), g_z2];

z3 = A2 * Theta2';
g_z3 = sigmoid(z3);
A3 = [ones(m,1), g_z3];

z4 = A3 * Theta3';
g_z4 = sigmoid(z4);
A4 = g_z4;

SquaredTheta1 = sum(sum(Theta1( : , 2:end).^2));
SquaredTheta2 = sum(sum(Theta2( : , 2:end).^2));
SquaredTheta3 = sum(sum(Theta3( : , 2:end).^2));
J = sum(sum((-yVec .* log(A4))-((1-yVec) .* log(1 - A4))))/m ...
    + (lambda*(SquaredTheta1+SquaredTheta2+SquaredTheta3))/(2*m);

    
d4 = A4 - yVec;
d4t3 = d4 * Theta3;

d3 = d4t3(:,2:end) .* sigmoidGradient(z3);
d3T2 = (d3 * Theta2);

d2 = d3T2(:,2:end) .* sigmoidGradient(z2);
    
D3 = (d4' * A3)/m;
D2 = (d3' * A2)/m;
D1 = (d2' * A1)/m;

Theta1_grad = D1;
Theta2_grad = D2;
Theta3_grad = D3;

RegTheta1 = (lambda/m)*Theta1;
RegTheta1(:, 1) = 0;
RegTheta2 = (lambda/m)*Theta2;
RegTheta2(:, 1) = 0;
RegTheta3 = (lambda/m)*Theta3;
RegTheta3(:, 1) = 0;

Theta1_grad = Theta1_grad + RegTheta1;
Theta2_grad = Theta2_grad + RegTheta2;
Theta3_grad = Theta3_grad + RegTheta3;

grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];
end
