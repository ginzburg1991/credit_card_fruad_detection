set.seed(5533)
# library(GPUTools)
# library(cudaBayesreg)
# library(gpuR)
library(cluster)
library(stats)
library(viridis)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(keras)
library(xgboost)
library(ROCR)
library(ROSE)
library(pROC)
library(ROSE)
library(tensorflow)
library(reticulate)
use_python("C:/Users/Benjamin/Documents/.virtualenvs/r-reticulate/Scripts/python.exe")
data <- read.csv("C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv")

splitIndex <- createDataPartition(data$fraud, p = .80, list = FALSE, times = 1)
trainData <- data[splitIndex,]
testData <- data[-splitIndex,]

model <- glm(fraud ~ ., data = trainData, family = "binomial")

predictions <- predict(model, testData, type = "response")
predictedClass <- ifelse(predictions > 0.5, 1, 0)

confusionMatrix <- table(Predicted = predictedClass, Actual = testData$fraud)
precision <- confusionMatrix[2, 2] / sum(confusionMatrix[2,])
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

roc_result <- roc(testData$fraud, as.numeric(predictions))
auc_value <- auc(roc_result)
print(paste("AUC:", auc_value))

model <- rpart(fraud ~ ., data = trainData, method = "class")
rpart.plot(model, main = "Decision Tree", extra = 102, digits = 5)
predictions <- predict(model, testData, type = "class")
confusionMatrix <- table(Predicted = predictions, Actual = testData$fraud)
print(confusionMatrix)

accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))

preProcValues <- preProcess(trainData, method = c("center", "scale"))
trainDataNorm <- predict(preProcValues, trainData)
testDataNorm <- predict(preProcValues, testData)
nn_model <- neuralnet(fraud ~ ., data = trainDataNorm, hidden = c(5), linear.output = FALSE)
predictions <- compute(nn_model, testDataNorm[, names(testDataNorm) != "fraud"])
predictedClass <- ifelse(predictions$net.result > 0.5, 1, 0)
confusionMatrix <- table(Predicted = predictedClass, Actual = testData$fraud)
print(confusionMatrix)

accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))

normalized_data <- scale(data[1:7])
indexes <- sample(1:nrow(normalized_data), size = 0.8 * nrow(normalized_data))
x_train <- normalized_data[indexes,]
y_train <- data$fraud[indexes]
x_test <- normalized_data[-indexes,]
y_test <- data$fraud[-indexes]

model <- keras_model_sequential()
model %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(7)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

model %>% evaluate(x_test, y_test)
predictions <- model %>% predict_classes(x_test)
