# Model Prediction Script
# Load required libraries
library(caret)
library(randomForest)

# Set working directory
setwd("c:/Users/Admin/Desktop/P2")

# Load the trained model
model <- readRDS("best_model.rds")

# Function to make predictions on new data
predict_loan_approval <- function(age, income, education, experience, credit_score) {
  # Create a data frame with the input
  new_data <- data.frame(
    age = age,
    income = income,
    education = education,
    experience = experience,
    credit_score = credit_score
  )
  
  # Make prediction
  prediction <- predict(model, new_data)
  probability <- predict(model, new_data, type = "prob")
  
  # Return results
  result <- list(
    prediction = as.character(prediction),
    probability_approved = probability[,2],
    probability_rejected = probability[,1]
  )
  
  return(result)
}

# Example predictions
cat("=== Example Predictions ===\n")

# Example 1: Young person with low income
result1 <- predict_loan_approval(25, 30000, 12, 2, 620)
cat("Example 1 - Age: 25, Income: 30000, Education: 12, Experience: 2, Credit Score: 620\n")
cat("Prediction:", result1$prediction, "\n")
cat("Probability of Approval:", round(result1$probability_approved, 3), "\n\n")

# Example 2: Middle-aged person with good credentials
result2 <- predict_loan_approval(40, 70000, 16, 12, 750)
cat("Example 2 - Age: 40, Income: 70000, Education: 16, Experience: 12, Credit Score: 750\n")
cat("Prediction:", result2$prediction, "\n")
cat("Probability of Approval:", round(result2$probability_approved, 3), "\n\n")

# Example 3: Older person with excellent credentials
result3 <- predict_loan_approval(50, 90000, 18, 20, 800)
cat("Example 3 - Age: 50, Income: 90000, Education: 18, Experience: 20, Credit Score: 800\n")
cat("Prediction:", result3$prediction, "\n")
cat("Probability of Approval:", round(result3$probability_approved, 3), "\n\n")

cat("To use this function for your own predictions, call:\n")
cat("predict_loan_approval(age, income, education, experience, credit_score)\n")