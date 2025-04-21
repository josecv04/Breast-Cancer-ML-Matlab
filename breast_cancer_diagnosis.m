
%% Data INPUT
%Here i bring in the data set from the link Yesh provided it is stored as a
%table since it is a csv file
data = readtable('data.csv');

%% Data FIltering 
% Here when I open up the csv file manually without calling it I notice
% there is a blank column at var 33 so I need to get rid of it also I need
% to change m and B into number so 0 or 1 as seen in previous homeworks
% also for sigmoid function later on maybe if yesh responds to email??
% only using mean data since it gets rid of outlier such as worst columns
new_matrix = data(:,1:12);
display(new_matrix)

%% Transforming M and B into 1 and 0
new_matrix.Var2 = categorical(new_matrix.Var2); % string to numerical value 
new_matrix.Var2 = double(new_matrix.Var2) - 1; % since it is letters they are treated as strings 
%to remove the string assignment we use double to give it a numerical
%value this assigns it by alphabetic order so b is 1 and M is 2 to convert
%this into 1 and 0 we just subtract by 1 

%% Describing the DATASET PART 1
% statistical description of the data includes minimum median and maximum 
disp('Dataset Summary:');
summary(new_matrix);
mean(new_matrix) % mean of each column
std(new_matrix) % standard deviation of each column 

%% Describing the DATASET PART 2
% this makes a bar graph to show the distribution of benign and malignant
% classifications 
figure;
histogram(new_matrix.Var2,'Binwidth',0.02);
title('Class Distribution');
xlabel('Diagnosis (0: Benign, 1: Malignant)');
ylabel('Count');

%% Describing the DATASET PART 3 (heatmap)
%the heat map plots the variation between all of the columns and the
%relationship between them this is done for all columns except the id and
%the tumor class
numericData = new_matrix{:, 3:end}; %This is the section 





correlationMatrix = corr(numericData);

% plotting the heat map using colors to determine the correlation strength
figure;
heatmap(new_matrix.Properties.VariableNames(3:end), ...
        new_matrix.Properties.VariableNames(3:end), ...
        correlationMatrix, ...
        'Colormap', jet, ...
        'ColorbarVisible', 'on');
title('Correlation Heatmap');


%% Seperating the data into the mean the se and the worst to visualize and
% understand what is going on in each category and its correlation 
% first will be the mean using plotmatrix
mean_data = new_matrix(:,3:12);
display(mean_data)
meandata2 = table2array(mean_data);
figure;
plotmatrix(meandata2);


%% clustering data prep (randomize data)

seed = rng(42); % searched this up apparently its for reproducibility 
randomrows = randperm(height(new_matrix)); %generates randome numbers 1 
%through 569 so I can assign the already existing rows a random new row
%value
randomized_data = new_matrix(randomrows,:); %Assigns existing rows a new 
randomized_data_array = table2array(randomized_data);
%value say row 1 now become row 324 
disp(randomized_data) % this is to check it 
randomized_important_variables = randomized_data(:,3:end);
disp(randomized_important_variables)

%% Splitting data into training and testing sets 80-20 split for all data
train = 0.8;
trainSize = round(train*height(randomized_data)); %this is
%for the rows so they are 80% of the data 
trainAllData = randomized_data(1:trainSize,:); % 455 rows w 10 col
testAllData=randomized_data(trainSize+1:end,:); % 569-455 = 114 rows with 10 col
size(trainAllData); %verify size is accurate
size(testAllData);

xtrainAll = table2array(trainAllData);
xtestAll = table2array(testAllData);

%% Splitting data into training and testing sets 80-20 split for relevant data
train = 0.8;
trainSize = round(train*height(randomized_important_variables)); %this is
%for the rows so they are 80% of the data 
trainData = randomized_important_variables(1:trainSize,:); % 455 rows w 10 col
testData=randomized_important_variables(trainSize+1:end,:); % 569-455 = 114 rows with 10 col
size(trainData); %verify size is accurate
size(testData);

xtrain = table2array(trainData);
xtest = table2array(testData);
%% Apply PCA for data set 
% first standardize the data  
xtraining = zscore(xtrain); %for training dataset
[coeff, score, ~, ~, explained] = pca(xtraining);

% Visualize the explained variance
figure;
bar(explained,0.2);
title('Explained Variance by Principal Components for training set');
xlabel('Principal Components');
ylabel('Variance Explained (%)');

% Plot PCA results (2D)
figure;
scatter(score(:, 1), score(:, 2), 20, trainAllData.Var2, "filled");
title('PCA Visualization for training dataset');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
colormap(jet);
colorbar;
%% Apply PCA for test set
xtesting = zscore(xtest); %for testing dataset

[coeff2, score2, ~, ~, explained2] = pca(xtesting);

% Visualize the explained variance
figure;
bar(explained2,0.2);
title('Explained Variance by Principal Components for testing set');
xlabel('Principal Components');
ylabel('Variance Explained (%)');

% Plot PCA results (2D)
figure;
scatter(score2(:, 1), score2(:, 2), 20, testAllData.Var2, "filled");
title('PCA Visualization for testing set');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
colormap(jet);
colorbar;

%% Apply K-Means clustering for training dataset
k = 2;
[idx, C]=kmeans(xtrain, k, 'distance', 'sqeuclidean', 'Replicates', 10);

figure;
scatter(xtrain(:, 1), xtrain(:, 2), 20, idx, 'filled');
title('K-Means Clustering for training set');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(jet);
colorbar;


%% Apply K-Means clustering for testing dataset
k = 2;
[idx2, C2]=kmeans(xtest, k, 'distance', 'sqeuclidean', 'Replicates', 10);

figure;
scatter(xtest(:, 1), xtest(:, 2), 20, idx2, 'filled');
title('K-Means Clustering for testing set');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(jet);
colorbar;

%% confusion matrix for training set (plot) and accuracy
% True labels for training set
trueLabelsTrain = trainAllData.Var2; % Assuming the second column corresponds to the labels (diagnosis)

% Map predicted cluster labels to true labels using majority voting
clusterMappingTrain = zeros(k, 1);
for i = 1:k
    clusterMappingTrain(i) = mode(trueLabelsTrain(idx == i));
end

% Map cluster indices to true labels
predictedLabelsTrain = arrayfun(@(x) clusterMappingTrain(x), idx);

% Confusion matrix for training
confMatrixTrain = confusionmat(trueLabelsTrain, predictedLabelsTrain);

% Calculate training accuracy
trainAccuracy = sum(predictedLabelsTrain == trueLabelsTrain) / length(trueLabelsTrain) * 100;

% Display training confusion matrix and accuracy
figure;
heatmap(confMatrixTrain, 'XData', {'Benign', 'Malignant'}, 'YData', {'Benign', 'Malignant'});
title('Confusion Matrix for Training Set');
xlabel('Predicted Labels');
ylabel('True Labels');
disp('Training Confusion Matrix:');
disp(confMatrixTrain);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);

%% confusion matrix for testing set (plot) and accuracy
% True labels for testing set
trueLabelsTest = testAllData.Var2; % Assuming the second column corresponds to the labels (diagnosis)

% Map predicted cluster labels to true labels using majority voting
clusterMappingTest = zeros(k, 1);
for i = 1:k
    clusterMappingTest(i) = mode(trueLabelsTest(idx2 == i));
end

% Map cluster indices to true labels
predictedLabelsTest = arrayfun(@(x) clusterMappingTest(x), idx2);

% Confusion matrix for testing
confMatrixTest = confusionmat(trueLabelsTest, predictedLabelsTest);

% Calculate testing accuracy
testAccuracy = sum(predictedLabelsTest == trueLabelsTest) / length(trueLabelsTest) * 100;

% Display testing confusion matrix and accuracy
figure;
heatmap(confMatrixTest, 'XData', {'Benign', 'Malignant'}, 'YData', {'Benign', 'Malignant'});
title('Confusion Matrix for Testing Set');
xlabel('Predicted Labels');
ylabel('True Labels');
disp('Testing Confusion Matrix:');
disp(confMatrixTest);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy);
%% Find the Optimal Number of PCA Components for Logistic Regression
maxComponents = size(score, 2); % Maximum number of PCA components available
bestNumComponents = 0;
highestAccuracy = 0;

% Loop through possible numbers of PCA components
for numComponents = 1:maxComponents
    % Train logistic regression model using selected PCA components
    mdlTrain = fitglm(score(:, 1:numComponents), trueLabelsTrain, ...
                      'Distribution', 'binomial', 'Link', 'logit');
    
    % Predict probabilities and classify for training set
    predictedProbTrain = predict(mdlTrain, score(:, 1:numComponents));
    predictedLabelsTrainLR = round(predictedProbTrain); % Threshold at 0.5

    % Calculate training accuracy for logistic regression
    trainAccuracyLR = sum(predictedLabelsTrainLR == trueLabelsTrain) / length(trueLabelsTrain) * 100;

    % Update the best number of components if accuracy improves
    if trainAccuracyLR > highestAccuracy
        highestAccuracy = trainAccuracyLR;
        bestNumComponents = numComponents;
    end
end

% Display the best number of PCA components and corresponding accuracy
fprintf('Best Number of PCA Components: %d\n', bestNumComponents);
fprintf('Highest Training Accuracy: %.2f%%\n', highestAccuracy);

%% Train and Evaluate Logistic Regression with Optimal Number of Components
% Train logistic regression using the optimal number of components
mdlTrainOptimal = fitglm(score(:, 1:bestNumComponents), trueLabelsTrain, ...
                         'Distribution', 'binomial', 'Link', 'logit');

% Training set evaluation
predictedProbTrainOptimal = predict(mdlTrainOptimal, score(:, 1:bestNumComponents));
predictedLabelsTrainOptimal = round(predictedProbTrainOptimal); % Threshold at 0.5

% Confusion matrix for logistic regression (training set)
confMatrixTrainOptimal = confusionmat(trueLabelsTrain, predictedLabelsTrainOptimal);

% Calculate training accuracy
trainAccuracyOptimal = sum(predictedLabelsTrainOptimal == trueLabelsTrain) / length(trueLabelsTrain) * 100;

% Display training confusion matrix and accuracy
figure;
heatmap(confMatrixTrainOptimal, 'XData', {'Benign', 'Malignant'}, 'YData', {'Benign', 'Malignant'});
title(sprintf('Logistic Regression Confusion Matrix for Training Set (%d PCA Components)', bestNumComponents));
xlabel('Predicted Labels');
ylabel('True Labels');
disp('Logistic Regression Optimal Training Confusion Matrix:');
disp(confMatrixTrainOptimal);
fprintf('Training Accuracy with Optimal Components: %.2f%%\n', trainAccuracyOptimal);

% Testing set evaluation
predictedProbTestOptimal = predict(mdlTrainOptimal, score2(:, 1:bestNumComponents));
predictedLabelsTestOptimal = round(predictedProbTestOptimal); % Threshold at 0.5

% Confusion matrix for logistic regression (testing set)
confMatrixTestOptimal = confusionmat(trueLabelsTest, predictedLabelsTestOptimal);

% Calculate testing accuracy
testAccuracyOptimal = sum(predictedLabelsTestOptimal == trueLabelsTest) / length(trueLabelsTest) * 100;

% Display testing confusion matrix and accuracy
figure;
heatmap(confMatrixTestOptimal, 'XData', {'Benign', 'Malignant'}, 'YData', {'Benign', 'Malignant'});
title(sprintf('Logistic Regression Confusion Matrix for Testing Set (%d PCA Components)', bestNumComponents));
xlabel('Predicted Labels');
ylabel('True Labels');
disp('Logistic Regression Optimal Testing Confusion Matrix:');
disp(confMatrixTestOptimal);
fprintf('Testing Accuracy with Optimal Components: %.2f%%\n', testAccuracyOptimal);
%% Plot for accuracy vs number of components for training set
maxComponents = size(score, 2); % Maximum number of PCA components available
trainAccuracies = zeros(maxComponents, 1); % Initialize array to store training accuracies

% Loop through possible numbers of PCA components
for numComponents = 1:maxComponents
    % Train logistic regression model using selected PCA components
    mdlTrain = fitglm(score(:, 1:numComponents), trueLabelsTrain, ...
                      'Distribution', 'binomial', 'Link', 'logit');
    
    % Predict probabilities and classify for training set
    predictedProbTrain = predict(mdlTrain, score(:, 1:numComponents));
    predictedLabelsTrainLR = round(predictedProbTrain); % Threshold at 0.5

    % Calculate training accuracy for logistic regression
    trainAccuracies(numComponents) = sum(predictedLabelsTrainLR == trueLabelsTrain) / length(trueLabelsTrain) * 100;
end

% Plot accuracy vs number of PCA components for training set
figure;
plot(1:maxComponents, trainAccuracies, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Accuracy vs Number of PCA Components (Training Set)', 'FontSize', 14);
xlabel('Number of PCA Components', 'FontSize', 12);
ylabel('Training Accuracy (%)', 'FontSize', 12);
grid on;
set(gca, 'XTick', 1:maxComponents); % Set x-axis ticks to integer values

%% Plot for accuracy vs number of components for testing set
testAccuracies = zeros(maxComponents, 1); % Initialize array to store testing accuracies

% Loop through possible numbers of PCA components
for numComponents = 1:maxComponents
    % Train logistic regression model using selected PCA components from training
    mdlTrain = fitglm(score(:, 1:numComponents), trueLabelsTrain, ...
                      'Distribution', 'binomial', 'Link', 'logit');
    
    % Predict probabilities and classify for testing set
    predictedProbTest = predict(mdlTrain, score2(:, 1:numComponents));
    predictedLabelsTestLR = round(predictedProbTest); % Threshold at 0.5

    % Calculate testing accuracy for logistic regression
    testAccuracies(numComponents) = sum(predictedLabelsTestLR == trueLabelsTest) / length(trueLabelsTest) * 100;
end

% Plot accuracy vs number of PCA components for testing set
figure;
plot(1:maxComponents, testAccuracies, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Accuracy vs Number of PCA Components (Testing Set)', 'FontSize', 14);
xlabel('Number of PCA Components', 'FontSize', 12);
ylabel('Testing Accuracy (%)', 'FontSize', 12);
grid on;
set(gca, 'XTick', 1:maxComponents); % Set x-axis ticks to integer values
