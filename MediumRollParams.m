function [Theta1, Theta2, Theta3] = MediumRollParams(trained_params, ...
                                                     Input_Layer_Size,...
                                                     Hidden_Layer_1_Size,...
                                                     Hidden_Layer_2_Size,...
                                                     Output_Layer_Size)

% Obtain Theta1 and Theta2 back from nn_params
t1r = Hidden_Layer_1_Size; t1c = Input_Layer_Size+1;
t2r = Hidden_Layer_2_Size; t2c = Hidden_Layer_1_Size+1;
t3r = Output_Layer_Size; t3c = Hidden_Layer_2_Size+1;
Theta1 = reshape(trained_params(1:t1r * t1c), t1r, t1c);
Theta2 = reshape(trained_params((1 + t1r * t1c):(t1r * t1c)+(t2r * t2c)), t2r, t2c);
Theta3 = reshape(trained_params((1 + (t1r * t1c)+(t2r * t2c)):end), t3r, t3c);

end