function [acc,pred] = Smallpredict(Theta1, Theta2, X, Y)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
% You need to return the following variables correctly 
pred = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[confidence, pred] = max(h2, [], 2);
pred=pred-1;
acc = mean(double(pred==Y))*100;
% =========================================================================


end
