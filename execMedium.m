function [trainingCost, validationCost, trainingAcc, validationAcc,...
          model_1_trained_params, Theta1, Theta2, Theta3]= execMedium(model_1_initial_params,...
                                                                      Input_Layer_Size,...
                                                                      Hidden_Layer_1_Size,...
                                                                      Hidden_Layer_2_Size,...
                                                                      Output_Layer_Size,...
                                                                      trainXR, trainY,...
                                                                      valdXR, valdY,...
                                                                      lambda, options)
%{
model_1_initial_params = BuildMediumModel(Input_Layer_Size,...
                                          Hidden_Layer_1_Size,...
                                          Hidden_Layer_2_Size,...
                                          Output_Layer_Size);
%}
model_1_trained_params = TrainMediumModel(model_1_initial_params,...
                                          Input_Layer_Size,...
                                          Hidden_Layer_1_Size,...
                                          Hidden_Layer_2_Size,...
                                          Output_Layer_Size,...
                                          trainXR, trainY,...
                                          lambda, options);

trainingCost = MediumnnCostFunction(model_1_trained_params, ...
                                   Input_Layer_Size,...
                                   Hidden_Layer_1_Size,...
                                   Hidden_Layer_2_Size,...
                                   Output_Layer_Size,...
                                   trainXR, trainY, lambda);
                               
validationCost = MediumnnCostFunction(model_1_trained_params, ...
                                      Input_Layer_Size,...
                                      Hidden_Layer_1_Size,...
                                      Hidden_Layer_2_Size,...
                                      Output_Layer_Size,...
                                      valdXR, valdY, 0);
                                 
[Theta1, Theta2, Theta3] = MediumRollParams(model_1_trained_params, ...
                                            Input_Layer_Size,...
                                            Hidden_Layer_1_Size,...
                                            Hidden_Layer_2_Size,...
                                            Output_Layer_Size);
                                       
[trainingAcc, train_pred] = Mediumpredict(Theta1, Theta2, Theta3,...
                                          trainXR, trainY);
[validationAcc, vald_pred] = Mediumpredict(Theta1, Theta2, Theta3,...
                                          valdXR, valdY);

end