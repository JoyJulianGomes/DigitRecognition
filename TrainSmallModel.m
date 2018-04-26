function [trained_params] = TrainSmallModel(initial_params_unrolled,...
                                            Input_Layer_Size,...
                                            Hidden_Layer_1_Size,...
                                            Output_Layer_Size,...
                                            trainXR, trainY,...
                                            lambda, options)
costFunction = @(p) SmallnnCostFunction(p, ...
                                        Input_Layer_Size, ...
                                        Hidden_Layer_1_Size, ...
                                        Output_Layer_Size,...
                                        trainXR, trainY, lambda);

[trained_params, cost_trail] = fmincg(costFunction, initial_params_unrolled, options);

end