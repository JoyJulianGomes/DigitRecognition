function [acc, pred] = Largepredict(Theta1, Theta2, Theta3, Theta4, X, Y)
m = size(X, 1);

h1 = sigmoid([ones(m, 1) X]  * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
h4 = sigmoid([ones(m, 1) h3] * Theta4');

[confidence, pred] = max(h4, [], 2);
pred=pred-1;%index starts from 1 but prediction has to be from 0-9
acc = mean(double(pred == Y)) * 100;
end
