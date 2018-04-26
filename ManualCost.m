%% Loading the MNIST dataset

fprintf("Loading Training, Validation, Testing data...\n");
load('MNIST/MNIST.mat');
fprintf("Loading Training, Validation, Testing data Complete\n");


%% Preparing data for neural network input

fprintf("Reshaping Train, Validation and Test Data...\n");
trainXR = reshape(trainX, [size(trainX, 1)*size(trainX, 2), size(trainX, 3)]);
trainXR = trainXR';
valdXR = reshape(valdX, [size(valdX, 1)*size(valdX, 2), size(valdX, 3)]);
valdXR = valdXR';
testXR = reshape(testX, [size(testX, 1)*size(testX, 2), size(testX, 3)]);
testXR = testXR';
fprintf("Reshaping Train and Test Data Complete\n");

m=size(trainXR, 1);

%% 
Input_Layer_Size    = 784; %Pixel input
Hidden_Layer_1_Size = 590; %784-193.5=590.5
Hidden_Layer_2_Size = 397; %784-193.5=397
Hidden_Layer_3_Size = 203; %397-193.5=203.5
Output_Layer_Size   = 10;
lambda = 0.32;
load('Result/M1_L0.32.mat');

M1_params_unrolled = [M1_Theta1(:); M1_Theta2(:);...
                      M1_Theta3(:); M1_Theta4(:)];
trainingCost = LargennCostFunction(M1_params_unrolled, ...
                                   Input_Layer_Size,...
                                   Hidden_Layer_1_Size,...
                                   Hidden_Layer_2_Size,...
                                   Hidden_Layer_3_Size,...
                                   Output_Layer_Size,...
                                   trainXR, trainY, lambda);
validationCost = LargennCostFunction(M1_params_unrolled, ...
                                     Input_Layer_Size,...
                                     Hidden_Layer_1_Size,...
                                     Hidden_Layer_2_Size,...
                                     Hidden_Layer_3_Size,...
                                     Output_Layer_Size,...
                                     valdXR, valdY, 0);