data = dir;
str = "BengaliDigitsResizedAndConverted";%"Dataf0lder";
fileStatus = 0;%train folder existance false
for i=1:size(data, 1)
    if(string(data(i).name) == str)
        data = data(i:i);
        fileStatus = 1;
        break;
    end
end
if fileStatus == 0
    fprintf('Data folder not found\n');
    return;
end

set = dir(strcat(data(1).folder, '\', data(1).name));
set = set(3:end);%<-contains Label Folders of EITHER test OR train folder
                 %3:end to avoid '.' and '..' file structure paths

numLabel = size(set, 1);
rawImageKJ = zeros(20, 20, 47, 10);%size(set, 1)==10=>true;
for j=1:10%numLabel
    fprintf('Label: %d\n',j);
    label = dir(strcat(set(j).folder, '\', set(j).name));
    label = label(3:end);%<-contains images of One of The Labels
    numImages = size(label,1);%size(label,1)==47=>true
    rawImageK = zeros(20, 20, 47);
        for k=1:47%numImages
            img = imread(strcat(label(k).folder,'\',label(k).name));
            rawImageK(:, :, k)= normalize(img);
        end
    rawImageKJ(:,:,:,j)=rawImageK;
end

imageNoLabel = reshape(rawImageKJ, 400, 47, 10);
%fig400 = figure;
%showImage(fig400, imageNoLabel);
%pause;

trainXSerialized = zeros(400, 420);
trainYSerialized = zeros(420,1);
elnum = 0;
for i=1:42
    for j=1:10
        elnum = elnum+1;
        trainXSerialized(:, elnum)=imageNoLabel(:, i, j);
        trainYSerialized(elnum,1) = j-1; 
    end
end
trainXSerialized= trainXSerialized';

testXSerialized = zeros(400, 50);
testYSerialized = zeros(50,1);
elnum = 0;
for i=42:46
    for j=1:10
        elnum = elnum+1;
        testXSerialized(:, elnum)=imageNoLabel(:, i, j);
        testYSerialized(elnum,1) = j-1; 
    end
end
testXSerialized = testXSerialized';

%Randomization of data

i = randperm(size(trainXSerialized,1));
trainX = trainXSerialized(i, :);
trainY = trainYSerialized(i, 1);

j = randperm(size(testXSerialized,1));
testX = testXSerialized(j, :);
testY = testYSerialized(j, 1);

save('BN_NUM_CHARS.mat', 'trainX', 'trainY', 'testX', 'testY');
%}
%{
data->|test-->|0->|im1    data.set    content.label label.id sample.name
              |   |im2        .content                             .path
              |
              |1->|im1
              |   |im2
              |
              |9->|im1
              |   |im2
              |
      |test-->|0->|im1
              |   |im2
              |
              |1->|im1
              |   |im2
              |
              |9->|im1
              |   |im2
%}