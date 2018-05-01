%% Clearing & Setup
close; clear; clc;
import mlreportgen.dom.*;
type='html-file';

%% Loading the MNIST dataset

fprintf("Loading Training, Validation, Testing data...\n");
load('Bengali_Dataset/BN_NUM_CHARS.mat');
fprintf("Loading Training, Validation, Testing data Complete\n");


%% Preparing data for neural network input

fprintf("Reshaping Train, Validation and Test Data...\n");
trainXR = trainX(1:336, :);
valdXR = trainX(337:end, :);% 20% validation set
testXR = testX;
valdY = trainY(337:end, :);
trainY = trainY(1:336, :);
fprintf("Reshaping Train and Test Data Complete\n");

m=size(trainXR, 1);

%% Building Models
options = optimset('MaxIter', 50);
lambda = [5.45];
epoch = 30;

%% Model-3: Small Equal Distance Model 784__H1__10

model = 3;
s=sprintf('Result/PerformanceReport-BN-Model-%i-3',model);
doc = Document(s, type);
table = createTable();

%(784-10)/2=387
%784 397 10
M3_Input_Layer_Size = 400;%Pixel input
M3_Hidden_Layer_1_Size = 195;%784-387=397
M3_Output_Layer_Size = 10;%Class output 397-387=10
for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildSmallModel(M3_Input_Layer_Size,M3_Hidden_Layer_1_Size,M3_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M3_trainingCost, M3_validationCost, M3_trainingAcc, M3_validationAcc,...
              nnparams, M3_Theta1, M3_Theta2] = execSmall(nnparams,...
                                                          M3_Input_Layer_Size,...
                                                          M3_Hidden_Layer_1_Size,...
                                                          M3_Output_Layer_Size,...
                                                          trainXR, trainY,...
                                                          valdXR, valdY,...
                                                          l, options);
         end
         AddRow(table, model, l, M3_trainingCost,M3_validationCost,M3_trainingAcc,M3_validationAcc);
         s=sprintf('M%i_L%1.2f.mat',model,l);
         save(s, 'M3_Theta1', 'M3_Theta2');
end
append(doc,table);
close(doc);
%rptview(doc.OutputPath);
%}

%% Model-6: Small  Exponential Model 784_H1_|_10

model = 6;
s=sprintf('Result/PerformanceReport-BN-Model-%i-3',model);
doc = Document(s, type);
table = createTable();

%(784-10)/3=258
M6_Input_Layer_Size = 400;%Pixel input
M6_Hidden_Layer_1_Size = 130;%784-258*1=526
M6_Output_Layer_Size = 10;%Class output 526-258*2=10
for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildSmallModel(M6_Input_Layer_Size,M6_Hidden_Layer_1_Size,M6_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M6_trainingCost, M6_validationCost, M6_trainingAcc, M6_validationAcc,...
              nnparams, M6_Theta1, M6_Theta2] = execSmall(nnparams,...
                                                          M6_Input_Layer_Size,...
                                                          M6_Hidden_Layer_1_Size,...
                                                          M6_Output_Layer_Size,...
                                                          trainXR, trainY,...
                                                          valdXR, valdY,...
                                                          l, options);
         end
         AddRow(table, model, l, M6_trainingCost,M6_validationCost,M6_trainingAcc,M6_validationAcc);
         s=sprintf('Result/BN/M%i_L%1.2f.mat',model,l);
         save(s, 'M6_Theta1', 'M6_Theta2');
end
append(doc,table);
close(doc);
%rptview(doc.OutputPath);