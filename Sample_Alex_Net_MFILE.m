% Clear all variables from the workspace
clear;

% Clear the Command Window
clc;

% Close all open figure windows
close all;

%% Select Image Dataset Directory

% Open a folder selection dialog and store the selected path
dataDir = uigetdir();

% Create an image datastore from the selected directory
% Include images from all subfolders and use folder names as labels
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Define the input image size required by the network
inputSize = [227 227];

% Resize every image when it is read from the datastore
imds.ReadFcn = @(filename)imresize(imread(filename), inputSize);

% Seed the random number generator using the current time
rng('shuffle');

% Split the dataset into 80% training and 20% validation data
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Create an image augmenter for data augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...      % Random horizontal flips
    'RandXTranslation', [-10 10], ... % Random horizontal shifts
    'RandYTranslation', [-10 10]);    % Random vertical shifts

% Create an augmented image datastore for training data
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);

% Create an augmented image datastore for validation data
% No augmentation is applied here
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% Define CNN Architecture (AlexNet-Inspired)

Layers = [

    % Input layer for RGB images
    imageInputLayer([227 227 3],"Name","imageinput")

    % First convolution layer
    convolution2dLayer([11 11],96,"Name","conv","Padding","same","Stride",[4 4])

    % ReLU activation
    reluLayer("Name","relu")

    % Cross-channel normalization layer
    crossChannelNormalizationLayer(5,"Name","crossnorm")

    % Max pooling layer
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same","Stride",[2 2])

    % Second convolution layer
    convolution2dLayer([5 5],256,"Name","conv_1","Padding","same","PaddingValue",2)

    % ReLU activation
    reluLayer("Name","relu_1")

    % Cross-channel normalization layer
    crossChannelNormalizationLayer(5,"Name","crossnorm_1")

    % Max pooling layer
    maxPooling2dLayer([3 3],"Name","maxpool_1","Padding","same","Stride",[2 2])

    % Third convolution layer
    convolution2dLayer([3 3],384,"Name","conv_2","Padding","same","PaddingValue",1)

    % ReLU activation
    reluLayer("Name","relu_2")

    % Fourth convolution layer
    convolution2dLayer([3 3],384,"Name","conv_3","Padding","same","PaddingValue",1)

    % ReLU activation
    reluLayer("Name","relu_3")

    % Fifth convolution layer
    convolution2dLayer([3 3],256,"Name","conv_4","Padding","same","PaddingValue",1)

    % ReLU activation
    reluLayer("Name","relu_4")

    % Final max pooling layer
    maxPooling2dLayer([3 3],"Name","maxpool_2","Padding","same","Stride",[2 2])

    % First fully connected layer
    fullyConnectedLayer(2048,"Name","fc")

    % ReLU activation
    reluLayer("Name","relu_5")

    % Dropout layer to reduce overfitting
    dropoutLayer(0.5,"Name","dropout")

    % Second fully connected layer
    fullyConnectedLayer(2048,"Name","fc_1")

    % ReLU activation
    reluLayer("Name","relu_6")

    % Dropout layer
    dropoutLayer(0.5,"Name","dropout_1")

    % Output layer with one neuron per class
    fullyConnectedLayer(numel(unique(imdsTrain.Labels)),"Name","fc_2")

    % Convert outputs to probabilities
    softmaxLayer("Name","softmax")

    % Classification output layer
    classificationLayer("Name", "output")
];

%% Notes / Future Improvements

% Ask about padding value of 1?
% Is cross-channel normalization equivalent to Local Response Normalization?
% Would batch normalization perform better?
% Investigate Alpha, Beta, K, and window size parameters
% Dataset source: Kaggle
% Test different mini-batch sizes (32, 64, 128, etc.)
% Example:
% crossChannelNormalizationLayer(5, 'Alpha', 1e-4, 'Beta', 0.75, 'K', 2)

%% Training Options

options = trainingOptions("sgdm", ...

    % Initial learning rate
    InitialLearnRate = 0.001,...

    % Number of complete passes through the dataset
    MaxEpochs = 10, ...

    % Number of images processed per mini-batch
    MiniBatchSize = 128, ...

    % Validation dataset
    ValidationData = imdsValidation, ...

    % Validate every 30 iterations
    ValidationFrequency = 30, ...

    % Display training progress plot
    Plots = "training-progress", ...

    % Suppress detailed command-window output
    Verbose = false);

% For multilabel classification:
% Metrics = "accuracy", ...
% AccuracyMetric = "multilabel", ...

%% Train the Network

% Train the CNN using the training datastore
net = trainNetwork(augimdsTrain, Layers, options);

%% Evaluate Performance

% Predict labels for validation images
YPred = classify(net, augimdsValidation);

% Get true validation labels
YValidation = imdsValidation.Labels;

% Compute classification accuracy
accuracy = mean(YPred == YValidation);

% Display validation accuracy as a percentage
disp(["Validation accuracy: ", accuracy*100 + "%"]);

% Create confusion matrix
confusionchart(YValidation, YPred, ...
    'Title', 'Confusion Matrix for Validation Set');

% Alternative confusion chart object creation
% chart = confusionchart(YValidation, YPred);

% Normalize rows of confusion matrix
chart.RowSummary = 'row-normalized';

% Normalize columns of confusion matrix
chart.ColumnSummary = 'column-normalized';

%% Alternative Image Preprocessing Methods

% Use a custom preprocessing function
% imds.ReadFcn = @(filename) preprocessImage(filename, inputSize);

% Custom preprocessing function
% function img = preprocessImage(filename, inputSize)
%
%     % Read image
%     img = imread(filename);
%
%     % Convert grayscale images to RGB
%     if size(img, 3) == 1
%         img = cat(3, img, img, img);
%     end
%
%     % Resize image
%     img = imresize(img, inputSize);
%
% end

% Alternative one-line grayscale-to-RGB conversion
% imds.ReadFcn = @(filename) ...
%     imresize(repmat(imread(filename), [1 1 3]), inputSize);
