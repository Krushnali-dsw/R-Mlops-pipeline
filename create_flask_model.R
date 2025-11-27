# Create a proper RandomForest model for Flask deployment
library(randomForest)

# Load dataset
data <- read.csv("dataset.csv")

# Prepare data
data$loan_approved <- as.factor(ifelse(data$loan_approved == 1, "approved", "denied"))

# Create training features
features <- c("age", "income", "education", "experience", "credit_score")
X <- data[, features]
y <- data$loan_approved

cat("📊 Creating pure RandomForest model for Flask deployment...\n")

# Train pure RandomForest model (not caret wrapper)
set.seed(123)
rf_model <- randomForest(
  x = X,
  y = y,
  ntree = 100,
  mtry = 2,
  importance = TRUE
)

# Print model info
print(rf_model)
print("Variable Importance:")
print(importance(rf_model))

# Save the pure RandomForest model
saveRDS(rf_model, "flask_random_forest.rds")

cat("✅ Pure RandomForest model saved as 'flask_random_forest.rds'\n")

# Test the model
cat("🧪 Testing the saved model...\n")
loaded_model <- readRDS("flask_random_forest.rds")

# Test prediction
test_sample <- data.frame(
  age = 35,
  income = 75000,
  education = 16,
  experience = 10,
  credit_score = 750
)

prediction <- predict(loaded_model, test_sample, type = "prob")
cat("Test prediction probabilities:\n")
print(prediction)

cat("Predicted class:", predict(loaded_model, test_sample), "\n")
cat("✅ Model testing successful!\n")