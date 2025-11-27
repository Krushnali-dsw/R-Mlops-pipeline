# Machine Learning Model Training Script
# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(pROC)

# Set working directory
setwd("c:/Users/Admin/Desktop/P2")

# Load and prepare the data
data <- read.csv("dataset.csv")
data$loan_approved <- factor(data$loan_approved, levels = c(0, 1), labels = c("Rejected", "Approved"))

# Set seed for reproducibility
set.seed(123)

# Split data into training and testing sets (80-20 split)
train_index <- createDataPartition(data$loan_approved, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# Prepare features and target
features <- c("age", "income", "education", "experience", "credit_score")
X_train <- train_data[, features]
y_train <- train_data$loan_approved
X_test <- test_data[, features]
y_test <- test_data$loan_approved

# Define cross-validation control
ctrl <- trainControl(method = "cv", number = 5, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = TRUE)

# Train different models
cat("\n=== Training Models ===\n")

# 1. Logistic Regression
cat("\nTraining Logistic Regression...\n")
model_logistic <- train(x = X_train, y = y_train,
                       method = "glm",
                       family = "binomial",
                       trControl = ctrl,
                       metric = "ROC")

# 2. Random Forest
cat("\nTraining Random Forest...\n")
model_rf <- train(x = X_train, y = y_train,
                  method = "rf",
                  trControl = ctrl,
                  metric = "ROC",
                  tuneLength = 3)

# 3. K-Nearest Neighbors
cat("\nTraining KNN...\n")
model_knn <- train(x = X_train, y = y_train,
                   method = "knn",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 3)

# Make predictions on test set
pred_logistic <- predict(model_logistic, X_test)
pred_rf <- predict(model_rf, X_test)
pred_knn <- predict(model_knn, X_test)

# Get prediction probabilities for ROC curves
prob_logistic <- predict(model_logistic, X_test, type = "prob")[,2]
prob_rf <- predict(model_rf, X_test, type = "prob")[,2]
prob_knn <- predict(model_knn, X_test, type = "prob")[,2]

# Calculate performance metrics
cat("\n=== Model Performance ===\n")

# Confusion matrices and accuracy
cm_logistic <- confusionMatrix(pred_logistic, y_test)
cm_rf <- confusionMatrix(pred_rf, y_test)
cm_knn <- confusionMatrix(pred_knn, y_test)

cat("\nLogistic Regression Accuracy:", cm_logistic$overall['Accuracy'], "\n")
cat("Random Forest Accuracy:", cm_rf$overall['Accuracy'], "\n")
cat("KNN Accuracy:", cm_knn$overall['Accuracy'], "\n")

# ROC curves and AUC
roc_logistic <- roc(y_test, prob_logistic)
roc_rf <- roc(y_test, prob_rf)
roc_knn <- roc(y_test, prob_knn)

cat("\nLogistic Regression AUC:", auc(roc_logistic), "\n")
cat("Random Forest AUC:", auc(roc_rf), "\n")
cat("KNN AUC:", auc(roc_knn), "\n")

# Feature importance for Random Forest
cat("\n=== Feature Importance (Random Forest) ===\n")
importance <- varImp(model_rf)
print(importance)

# Plot ROC curves
png("roc_curves.png", width = 800, height = 600)
plot(roc_logistic, col = "blue", main = "ROC Curves Comparison")
plot(roc_rf, add = TRUE, col = "red")
plot(roc_knn, add = TRUE, col = "green")
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "KNN"),
       col = c("blue", "red", "green"), lwd = 2)
dev.off()

# Plot feature importance
png("feature_importance.png", width = 800, height = 600)
plot(importance, main = "Feature Importance (Random Forest)")
dev.off()

# Save the best model (based on AUC)
aucs <- c(auc(roc_logistic), auc(roc_rf), auc(roc_knn))
best_model_idx <- which.max(aucs)
model_names <- c("Logistic Regression", "Random Forest", "KNN")
models <- list(model_logistic, model_rf, model_knn)

best_model <- models[[best_model_idx]]
best_model_name <- model_names[best_model_idx]

cat("\nBest performing model:", best_model_name, "with AUC:", max(aucs), "\n")

# Save the best model
saveRDS(best_model, file = "best_model.rds")
cat("Best model saved as 'best_model.rds'\n")

# Print detailed results for the best model
cat("\n=== Best Model Details ===\n")
print(best_model)

cat("\n=== Training Complete ===\n")
cat("Generated files:\n")
cat("- best_model.rds (saved model)\n")
cat("- roc_curves.png\n")
cat("- feature_importance.png\n")