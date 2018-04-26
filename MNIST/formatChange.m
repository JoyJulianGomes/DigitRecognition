%% Original UByte format to mat format conversion
[trainX, trainY] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 50000, 0);
%50000 tarining data loaded
[valdX, valdY] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 10000, 50000);%100 tarining data loaded
%10000 tarining data loaded
[testX, testY] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);%100 test data loaded
%10000 testing data loaded

save('MNIST.mat', 'trainX', 'trainY', 'valdX', 'valdY', 'testX', 'testY');