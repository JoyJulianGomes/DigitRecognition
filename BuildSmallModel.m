function initial_params_unrolled = BuildSmallModel(Input_Layer_Size,...
                                                   Hidden_Layer_1_Size,...
                                                   Output_Layer_Size)
%This large Model Genre has 3 hidden layers
%Therefore it will have 4 weight matrix
initial_Theta_1 = randInitializeWeights(Input_Layer_Size, Hidden_Layer_1_Size);
initial_Theta_2 = randInitializeWeights(Hidden_Layer_1_Size, Output_Layer_Size);

% Unroll parameters to use in optimizer function
initial_params_unrolled = [initial_Theta_1(:); initial_Theta_2(:)];                     
end