rootFolder = fullfile('Images');

categories = {'CT_COVID', 'CT_NonCOVID'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldername');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds);

CT_COVID = find(imds.Labels == 'CT_COVID', 1);
CT_NonCOVID = find(imds.Labels == 'CT_NonCOVID', 1);

figure 
subplot(2,1,1);
imshow(readimage(imds, CT_COVID));
subplot(2,1,2);
imshow(readimage(imds, CT_NonCOVID));

net = resnet50();

net.Layers(1)
net.Layers(end)

numel(net.Layers(end).ClassNames)
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize, ...
    trainingSet, 'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, ...
    testSet, 'ColorPreprocessing', 'gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

figure
montage(w1)
title('First Convolutional Layer Weight')

featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLables = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLables, ...
    'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testLables = testSet.Labels;
confMat = confusionmat(testLables, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));

mean(diag(confMat));


