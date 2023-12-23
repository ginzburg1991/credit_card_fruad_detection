library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(pROC)
library(dplyr)
library(pROC)
set.seed(5533)
dataset <- read.csv("/Users/benjaminlindeen/developement/credit-card-fraud-detection/datatset/card_transdata.csv")

dataset <- na.omit(dataset)
small_constant <- 0.0000001
dataset <- dataset %>%
  mutate(distance_from_home_log = log(ifelse(distance_from_home <= 0, small_constant, distance_from_home)),
         distance_from_last_transaction_log = log(ifelse(distance_from_last_transaction <= 0, small_constant, distance_from_last_transaction)),
         ratio_to_median_purchase_price_log = log(ifelse(ratio_to_median_purchase_price <= 0, small_constant, ratio_to_median_purchase_price)))

hist(dataset$distance_from_home_log, main = "Histogram of Log Transformed Distance from Home", xlab = "Log Distance from Home")
hist(dataset$distance_from_last_transaction_log, main = "Histogram of Log Transformed Distance from Last Transaction", xlab = "Log Distance from Last Transaction")
hist(dataset$ratio_to_median_purchase_price_log, main = "Histogram of Log Transformed Ratio to Median Purchase Price", xlab = "Log Ratio to Median Purchase Price")

dataset <- dataset %>%
  select(-distance_from_home, -distance_from_last_transaction, -ratio_to_median_purchase_price)

partition <- createDataPartition(dataset$fraud, p = 0.8, list = FALSE)
training <- dataset[partition,]
testing <- dataset[-partition,]

class_weights <- ifelse(training$fraud == 1, (1 / table(training$fraud)[2]), (1 / table(training$fraud)[1]))

lin_model <- lm(fraud ~ ., data = training, weights = class_weights)
summary(lin_model)
predictions_lin <- predict(lin_model, testing)
predicted_classes_lin <- ifelse(predictions_lin > 0.5, 1, 0)
confusionMatrix(factor(predicted_classes_lin), factor(testing$fraud))

conf_matrix_lin <- confusionMatrix(factor(predicted_classes_lin), factor(testing$fraud))
precision_lin <- conf_matrix_lin$byClass['Pos Pred Value']
recall_lin <- conf_matrix_lin$byClass['Sensitivity']
f1_score_lin <- 2 * precision_lin * recall_lin / (precision_lin + recall_lin)

print(paste("Linear Regression - Precision:", precision_lin))
print(paste("Linear Regression - Recall:", recall_lin))
print(paste("Linear Regression - F1 Score:", f1_score_lin))

log_model <- glm(fraud ~ ., data = training, family = binomial(), weights = class_weights)
predictions <- predict(log_model, testing, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(factor(predictions), factor(testing$fraud))

conf_matrix_log <- confusionMatrix(factor(predictions), factor(testing$fraud))
precision_log <- conf_matrix_log$byClass['Pos Pred Value']
recall_log <- conf_matrix_log$byClass['Sensitivity']
f1_score_log <- 2 * precision_log * recall_log / (precision_log + recall_log)

print(paste("Logistic Regression - Precision:", precision_log))
print(paste("Logistic Regression - Recall:", recall_log))
print(paste("Logistic Regression - F1 Score:", f1_score_log))

tree_model <- rpart(fraud ~ ., data = training, method = "class", weights = class_weights)
predictions <- predict(tree_model, newdata = testing, type = "class")
predictions_factor <- factor(predictions, levels = c(0, 1))
testing_fraud_factor <- factor(testing$fraud, levels = c(0, 1))
rpart.plot(tree_model, type = 3, box.palette = "RdBu", shadow.col = "gray", branch = 1, extra = 102, under = TRUE, cex = 0.6, tweak = 1.2)
confusionMatrix(predictions_factor, testing_fraud_factor)

conf_matrix_tree <- confusionMatrix(predictions_factor, testing_fraud_factor)
precision_tree <- conf_matrix_tree$byClass['Pos Pred Value']
recall_tree <- conf_matrix_tree$byClass['Sensitivity']
f1_score_tree <- 2 * precision_tree * recall_tree / (precision_tree + recall_tree)

print(paste("Decision Tree - Precision:", precision_tree))
print(paste("Decision Tree - Recall:", recall_tree))
print(paste("Decision Tree - F1 Score:", f1_score_tree))
