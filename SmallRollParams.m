function [Theta1, Theta2] = SmallRollParams(trained_params, ...
                                                     Input_Layer_Size,...
                                                     Hidden_Layer_1_Size,...
                                                     Output_Layer_Size)

% Obtain Theta1 and Theta2 back from nn_params
t1r = Hidden_Layer_1_Size; t1c = Input_Layer_Size+1;
t2r = Output_Layer_Size; t2c = Hidden_Layer_1_Size+1;
Theta1 = reshape(trained_params(1:t1r * t1c), t1r, t1c);
Theta2 = reshape(trained_params((1 + t1r * t1c):end), t2r, t2c);
end