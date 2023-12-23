# Load libraries
library(caret)
library(ggplot2)
library(smotefamily)
library(rpart)
library(rpart.plot)
library(xgboost)
library(ggcorrplot)
library(pROC)
dataset <- read.csv("C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv")

summary(dataset)
head(dataset)
dim(dataset)
ggplot(dataset, aes(x = distance_from_home)) + geom_histogram(binwidth = 10, fill = "blue", color = "black")
ggplot(dataset, aes(x = as.factor(fraud))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution in Dataset", x = "Fraud", y = "Count")

dataset <- na.omit(dataset)

set.seed(123)
partition <- createDataPartition(dataset$fraud, p = 0.8, list = FALSE)
training <- dataset[partition,]
testing <- dataset[-partition,]

features <- training[, -ncol(training)]
labels <- training$fraud

count_non_fraud <- nrow(training[training$fraud == 0,])
print(count_non_fraud)

count_fraud <- nrow(training[training$fraud == 1,])
print(count_fraud)

dup_size <- count_non_fraud / count_fraud
print(dup_size)

smote_data <- SMOTE(features, labels, K = 5, dup_size = 1)

synthetic_samples1 <- smote_data$data
synthetic_samples1 <- synthetic_samples1[, !names(synthetic_samples1) %in% "class"]
synthetic_samples1$fraud <- rep(1, nrow(synthetic_samples1))

training_features1 <- training[, -ncol(training)]
training_features1$fraud <- training$fraud
training_balanced1 <- rbind(training_features1, synthetic_samples1)

table(training_balanced1$fraud)
