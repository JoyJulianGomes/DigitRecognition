%% Clearing & Setup
close; clear; clc;
import mlreportgen.dom.*;
type='html-file';

%% Loading the MNIST dataset

fprintf("Loading Training, Validation, Testing data...\n");
load('Bengali_Dataset/BN_NUM_CHARS.mat');
%load('MNIST/MNIST.mat');
fprintf("Loading Training, Validation, Testing data Complete\n");


%% Preparing data for neural network input

%fprintf("Reshaping Train, Validation and Test Data...\n");
%trainXR = reshape(trainX, [size(trainX, 1)*size(trainX, 2), size(trainX, 3)]);
%trainXR = trainXR';
trainXR = trainX;
%valdXR = reshape(valdX, [size(valdX, 1)*size(valdX, 2), size(valdX, 3)]);
%valdXR = valdXR';
valdXR = trainXR(:
testXR = reshape(testX, [size(testX, 1)*size(testX, 2), size(testX, 3)]);
testXR = testXR';
fprintf("Reshaping Train and Test Data Complete\n");

m=size(trainXR, 1);

%% Building Models
options = optimset('MaxIter', 50);
lambda = [0.16, 0.32, 0.64, 1.28 2.56 5.12 10.24];
epoch = 5;
         
%% Model-1: Large Equal Distance Model 784__H1__H2__H3__10
%{
model=1;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/4=193.5
%784 590 397 203 10
M1_Input_Layer_Size    = 784; %Pixel input
M1_Hidden_Layer_1_Size = 590; %784-193.5=590.5
M1_Hidden_Layer_2_Size = 397; %784-193.5=397
M1_Hidden_Layer_3_Size = 203; %397-193.5=203.5
M1_Output_Layer_Size   = 10;  %Class output 203.5-193.5=10

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildLargeModel(M1_Input_Layer_Size,...
                               M1_Hidden_Layer_1_Size,...
                               M1_Hidden_Layer_2_Size,...
                               M1_Hidden_Layer_3_Size,...
                               M1_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M1_trainingCost, M1_validationCost, M1_trainingAcc, M1_validationAcc,...
              nnparams, M1_Theta1, M1_Theta2, M1_Theta3, M1_Theta4] = execLarge(nnparams,...
                                                                                M1_Input_Layer_Size,...
                                                                                M1_Hidden_Layer_1_Size,...
                                                                                M1_Hidden_Layer_2_Size,...
                                                                                M1_Hidden_Layer_3_Size,...
                                                                                M1_Output_Layer_Size,...
                                                                                trainXR, trainY,...
                                                                                valdXR, valdY,...
                                                                                l, options);
         end
         AddRow(table, model, l, M1_trainingCost, M1_validationCost, M1_trainingAcc, M1_validationAcc);
         s=sprintf('Result/M%i_L%1.2f.mat',model,l);
         save(s, 'M1_Theta1', 'M1_Theta2', 'M1_Theta3', 'M1_Theta4');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-2: Medium Equal Distance Model 784__H1__H2__10

model=2;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/3=258
%784 526 268 10
M2_Input_Layer_Size = 784;%Pixel input
M2_Hidden_Layer_1_Size = 526;%784-258=526
M2_Hidden_Layer_2_Size = 268;%526-258=268
M2_Output_Layer_Size = 10;%Class output 268-258=10

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildMediumModel(M2_Input_Layer_Size,...
                                M2_Hidden_Layer_1_Size,...
                                M2_Hidden_Layer_2_Size,...
                                M2_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M2_trainingCost, M2_validationCost, M2_trainingAcc, M2_validationAcc,...
              nnparams, M2_Theta1, M2_Theta2, M2_Theta3] = execMedium(nnparams,...
                                                                      M2_Input_Layer_Size,...
                                                                      M2_Hidden_Layer_1_Size,...
                                                                      M2_Hidden_Layer_2_Size,...
                                                                      M2_Output_Layer_Size,...
                                                                      trainXR, trainY,...
                                                                      valdXR, valdY,...
                                                                      l, options);
         end
         AddRow(table, model, l, M2_trainingCost, M2_validationCost, M2_trainingAcc, M2_validationAcc);
         s=sprintf('Result/M%i_L%1.2f.mat',model,l);
         save(s, 'M2_Theta1', 'M2_Theta2', 'M2_Theta3');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);

%% Model-3: Small Equal Distance Model 784__H1__10
%{
model = 3;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/2=387
%784 397 10
M3_Input_Layer_Size = 784;%Pixel input
M3_Hidden_Layer_1_Size = 397;%784-387=397
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
rptview(doc.OutputPath);
%}
%% Model-4: Large  Exponential Model 784_H1_|_H2_|_|_|_H3_|_|_|_|_|_|_|_10
%{
model=4;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

M4_Input_Layer_Size = 784;%Pixel input
M4_Hidden_Layer_1_Size = 732;%784-51.6*1=732.4
M4_Hidden_Layer_2_Size = 629;%732.4-51.6*2=629.2
M4_Hidden_Layer_3_Size = 423;%397-51.6*4=422.8
M4_Output_Layer_Size = 10;%Class output 422.8-51.6*8=10

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildLargeModel(M4_Input_Layer_Size,...
                               M4_Hidden_Layer_1_Size,...
                               M4_Hidden_Layer_2_Size,...
                               M4_Hidden_Layer_3_Size,...
                               M4_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M4_trainingCost, M4_validationCost, M4_trainingAcc, M4_validationAcc,...
              nnparams, M4_Theta1, M4_Theta2, M4_Theta3, M4_Theta4] = execLarge(nnparams,...
                                                                                M4_Input_Layer_Size,...
                                                                                M4_Hidden_Layer_1_Size,...
                                                                                M4_Hidden_Layer_2_Size,...
                                                                                M4_Hidden_Layer_3_Size,...
                                                                                M4_Output_Layer_Size,...
                                                                                trainXR, trainY,...
                                                                                valdXR, valdY,...
                                                                                l, options);
         end
         AddRow(table, model, l, M4_trainingCost, M4_validationCost, M4_trainingAcc, M4_validationAcc);
         s=sprintf('Result/M%i_L0%1.2f.mat',model,l);
         save(s, 'M4_Theta1', 'M4_Theta2', 'M4_Theta3', 'M4_Theta4');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-5: Medium Exponential Model 784_H1_|_H2_|_|_|_10
%{
model=5;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/7=110.57
M5_Input_Layer_Size = 784;%Pixel input
M5_Hidden_Layer_1_Size = 673;%784-110.57*1=673.43
M5_Hidden_Layer_2_Size = 452;%673.43-110.57*2=452.29
M5_Output_Layer_Size = 10;%Class output 452.29-110.57*4=10.01

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildMediumModel(M5_Input_Layer_Size,...
                                M5_Hidden_Layer_1_Size,...
                                M5_Hidden_Layer_2_Size,...
                                M5_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M5_trainingCost, M5_validationCost, M5_trainingAcc, M5_validationAcc,...
              nnparams, M5_Theta1, M5_Theta2, M5_Theta3] = execMedium(nnparams,...
                                                                      M5_Input_Layer_Size,...
                                                                      M5_Hidden_Layer_1_Size,...
                                                                      M5_Hidden_Layer_2_Size,...
                                                                      M5_Output_Layer_Size,...
                                                                      trainXR, trainY,...
                                                                      valdXR, valdY,...
                                                                      l, options);
         end
         AddRow(table, model, l, M5_trainingCost, M5_validationCost, M5_trainingAcc, M5_validationAcc);
         s=sprintf('Result/M%i_L%1.2f.mat',model,l);
         save(s, 'M5_Theta1', 'M5_Theta2', 'M5_Theta3');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-6: Small  Exponential Model 784_H1_|_10
%{
model = 6;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/3=258
M6_Input_Layer_Size = 784;%Pixel input
M6_Hidden_Layer_1_Size = 526;%784-258*1=526
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
         s=sprintf('M%i_L%1.2f.mat',model,l);
         save(s, 'M6_Theta1', 'M6_Theta2');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-7: Large  Exponential Model Reverse 784_|_|_|_|_|_|_|_H1_|_|_|_H2_|_H3_10
%{
model=7;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/15=51.6
M7_Input_Layer_Size = 784;%Pixel input
M7_Hidden_Layer_1_Size = 371;%784.0-51.6*8=371.2
M7_Hidden_Layer_2_Size = 164;%371.2-51.6*4=164.8
M7_Hidden_Layer_3_Size = 61; %164.8-51.6*2=061.6
M7_Output_Layer_Size = 10;%Class output 61.6-51.6*1=10

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildLargeModel(M7_Input_Layer_Size,...
                               M7_Hidden_Layer_1_Size,...
                               M7_Hidden_Layer_2_Size,...
                               M7_Hidden_Layer_3_Size,...
                               M7_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M7_trainingCost, M7_validationCost, M7_trainingAcc, M7_validationAcc,...
              nnparams, M7_Theta1, M7_Theta2, M7_Theta3, M7_Theta4] = execLarge(nnparams,...
                                                                                M7_Input_Layer_Size,...
                                                                                M7_Hidden_Layer_1_Size,...
                                                                                M7_Hidden_Layer_2_Size,...
                                                                                M7_Hidden_Layer_3_Size,...
                                                                                M7_Output_Layer_Size,...
                                                                                trainXR, trainY,...
                                                                                valdXR, valdY,...
                                                                                l, options);
         end
         AddRow(table, model, l, M7_trainingCost, M7_validationCost, M7_trainingAcc, M7_validationAcc);
         s=sprintf('Result/M%i_L0%1.2f.mat',model,l);
         save(s, 'M7_Theta1', 'M7_Theta2', 'M7_Theta3', 'M7_Theta4');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-8: Medium Exponential Model Reverse 784_|_|_|_H1_|_H2_10
%{
model=8;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();


%(784-10)/7=110.57
M8_Input_Layer_Size = 784;%Pixel input
M8_Hidden_Layer_1_Size = 342;%784.00-110.57*4=341.72
M8_Hidden_Layer_2_Size = 121;%341.72-110.57*2=120.58
M8_Output_Layer_Size = 10;%Class output 120.58-110.57*1=10.01

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildMediumModel(M8_Input_Layer_Size,...
                                M8_Hidden_Layer_1_Size,...
                                M8_Hidden_Layer_2_Size,...
                                M8_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M8_trainingCost, M8_validationCost, M8_trainingAcc, M8_validationAcc,...
              nnparams, M8_Theta1, M8_Theta2, M8_Theta3] = execMedium(nnparams,...
                                                                      M8_Input_Layer_Size,...
                                                                      M8_Hidden_Layer_1_Size,...
                                                                      M8_Hidden_Layer_2_Size,...
                                                                      M8_Output_Layer_Size,...
                                                                      trainXR, trainY,...
                                                                      valdXR, valdY,...
                                                                      l, options);
         end
         AddRow(table, model, l, M8_trainingCost, M8_validationCost, M8_trainingAcc, M8_validationAcc);
         s=sprintf('Result/M%i_L%1.2f.mat',model,l);
         save(s, 'M8_Theta1', 'M8_Theta2', 'M8_Theta3');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}
%% Model-9: Small  Exponential Model Reverse 784_|_H1_10
%{
model = 9;
s=sprintf('Result/PerformanceReport-Model-%i',model);
doc = Document(s, type);
table = createTable();

%(784-10)/3=258
M9_Input_Layer_Size = 784;%Pixel input
M9_Hidden_Layer_1_Size = 268;%784-258*2=268
M9_Output_Layer_Size = 10;%Class output 526-258*1=10

for i=1:size(lambda,2)
    l = lambda(i);
    nnparams = BuildSmallModel(M9_Input_Layer_Size,M9_Hidden_Layer_1_Size,M9_Output_Layer_Size);
         for j = 1:epoch
             fprintf("Model: %d Lambda: %2.2f Epoch: %d\n", model, l, j);
             [M9_trainingCost, M9_validationCost, M9_trainingAcc, M9_validationAcc,...
              nnparams, M9_Theta1, M9_Theta2] = execSmall(nnparams,...
                                                          M9_Input_Layer_Size,...
                                                          M9_Hidden_Layer_1_Size,...
                                                          M9_Output_Layer_Size,...
                                                          trainXR, trainY,...
                                                          valdXR, valdY,...
                                                          l, options);
         end
         AddRow(table, model, l, M9_trainingCost,M9_validationCost,M9_trainingAcc,M9_validationAcc);
         s=sprintf('M%i_L%1.2f.mat',model,l);
         save(s, 'M9_Theta1', 'M9_Theta2');
end
append(doc,table);
close(doc);
rptview(doc.OutputPath);
%}