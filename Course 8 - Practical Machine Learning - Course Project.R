# Coursera John Hopkinds Data Science Specialization
# Course 8: Practical Machine Learning
# Course 8 Project: predict quality of exercise movement; goal is to predict 'classe' variable

# Clear workspace
rm(list = ls())

# Load libraries
library(caret)
library(AppliedPredictiveModeling)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)

# Set working directory and read in data
setwd("~/R/Data")
pml_training <- read.csv("pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
pml_testing <- read.csv("pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))

pml_training <- as.data.frame(pml_training)
pml_testing <- as.data.frame(pml_testing)

# Examine data
# str(pml_training)
# str(pml_testing)
# head(pml_training)
# head(pml_testing)
# summary(pml_training$classe)
# summary(pml_testing$classe)
which(colnames(pml_training) == "classe") # column 160 has 'classe' variable
which(colnames(pml_testing) == "classe") # column 160 has 'classe' variable

# Clean training data - remove empty columns
pml_training <- pml_training[,(colSums(is.na(pml_training)) == 0)]
pml_testing <- pml_testing[,(colSums(is.na(pml_testing)) == 0)]
dim(pml_training)
dim(pml_testing)
# Clean training data - remove values that are very close to zero
nzv_pml_training <- nearZeroVar(pml_training,saveMetrics=TRUE)
pml_training <- pml_training[,nzv_pml_training$nzv==FALSE]
nzv_pml_testing <- nearZeroVar(pml_testing,saveMetrics=TRUE)
pml_testing <- pml_testing[,nzv_pml_testing$nzv==FALSE]
dim(pml_training)
dim(pml_testing)

# Partition pml_training into training (60%) and testing (40%) sets
inTrain <- createDataPartition(y=pml_training$classe, p=0.6, list=FALSE)
my_train <- pml_training[inTrain, ]
my_test <- pml_training[-inTrain, ]
dim(my_train)
dim(my_test)


# Start with basic classification tree
tree_model <- rpart(classe ~ ., data = my_train, method = "class")
fancyRpartPlot(tree_model)
pred_tree_model <- predict(tree_model, my_test, type = "class")
confusionMatrix(pred_tree_model, my_test$classe)
1 - postResample(my_test$classe, pred_tree_model)[[1]] # out of sample accuracy

# Try random forest
rf_model1 <- randomForest(classe ~ ., data = my_train)
rf_model1
pred_rf_model1 <- predict(rf_model1, my_test, type = "class")
confusionMatrix(pred_rf_model1, my_test$classe)
1 - postResample(my_test$classe, pred_rf_model1)[[1]] # out of sample accuracy

# Alternate method random forest
rf_model2 <- train(classe ~.,  data = my_train, method = "rf", prox = TRUE, ntree = 10)
rf_model2
pred_rf_model2 <- predict(rf_model2, my_test, type = "raw")
confusionMatrix(pred_rf_model2, my_test$classe)
1 - postResample(my_test$classe, pred_rf_model2)[[1]] # out of sample accuracy

# High accuracy leads to suspicion of overfitting
# Use k-fold cross-validation to avoid overfitting
rf_cv_model <- train(classe ~ ., method = "rf", data = my_train, trControl = trainControl(method='cv'), number = 10, allowParallel = TRUE, importance = TRUE)
rf_cv_model
pred_rf_cv_model <- predict(rf_cv_model, my_test, type = "raw")
confusionMatrix(pred_rf_cv_model, my_test$classe)
1 - postResample(my_test$classe, pred_rf_cv_model)[[1]] # out of sample accuracy

# Prediction using test data
# prediction_final_test <- predict(tree_model, pml_testing, type = "class")
# prediction_final_test <- predict(rf_model2, pml_testing, type = "raw")
prediction_final_test <- predict(rf_cv_model, pml_testing, type = "raw")
prediction_final_test

