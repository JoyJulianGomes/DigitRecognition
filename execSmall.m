function [trainingCost, validationCost, trainingAcc, validationAcc,...
          model_1_trained_params, Theta1, Theta2]= execSmall(model_1_initial_params,...
                                     Input_Layer_Size,...
                                     Hidden_Layer_1_Size,...
                                     Output_Layer_Size,...
                                     trainXR, trainY,...
                                     valdXR, valdY,...
                                     lambda, options)
%{
ignoring this part as nnparams need to send back and forth in epoch setting
model_1_initial_params = BuildSmallModel(Input_Layer_Size,...
                                          Hidden_Layer_1_Size,...
                                          Output_Layer_Size);
%}
model_1_trained_params = TrainSmallModel(model_1_initial_params,...
                                          Input_Layer_Size,...
                                          Hidden_Layer_1_Size,...
                                          Output_Layer_Size,...
                                          trainXR, trainY,...
                                          lambda, options);

trainingCost = SmallnnCostFunction(model_1_trained_params, ...
                                   Input_Layer_Size,...
                                   Hidden_Layer_1_Size,...
                                   Output_Layer_Size,...
                                   trainXR, trainY, lambda);
                               
validationCost = SmallnnCostFunction(model_1_trained_params, ...
                                      Input_Layer_Size,...
                                      Hidden_Layer_1_Size,...
                                      Output_Layer_Size,...
                                      valdXR, valdY, 0);
                                 
[Theta1, Theta2] = SmallRollParams(model_1_trained_params, ...
                                            Input_Layer_Size,...
                                            Hidden_Layer_1_Size,...
                                            Output_Layer_Size);
                                       
[trainingAcc, train_pred] = Smallpredict(Theta1, Theta2,...
                                          trainXR, trainY);
[validationAcc, vald_pred] = Smallpredict(Theta1, Theta2,...
                                          valdXR, valdY);

end