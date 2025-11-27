# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(corrplot)
library(ggplot2)

# Set working directory (adjust if needed)
setwd("c:/Users/Admin/Desktop/P2")

# Load the dataset
data <- read.csv("dataset.csv")

# Display basic information about the dataset
cat("Dataset Shape:", dim(data), "\n")
cat("Column Names:", colnames(data), "\n")

# Check for missing values
cat("Missing values per column:\n")
print(colSums(is.na(data)))

# Display summary statistics
cat("\nSummary Statistics:\n")
print(summary(data))

# Check the distribution of the target variable
cat("\nTarget Variable Distribution:\n")
print(table(data$loan_approved))

# Convert target variable to factor
data$loan_approved <- as.factor(data$loan_approved)

# Create correlation matrix for numeric variables
numeric_vars <- data[sapply(data, is.numeric)]
correlation_matrix <- cor(numeric_vars)

# Plot correlation matrix
png("correlation_matrix.png", width = 800, height = 600)
corrplot(correlation_matrix, method = "circle", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black")
dev.off()

# Create some exploratory plots
png("age_income_plot.png", width = 800, height = 600)
ggplot(data, aes(x = age, y = income, color = loan_approved)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Age vs Income by Loan Approval Status",
       x = "Age", y = "Income", color = "Loan Approved") +
  theme_minimal()
dev.off()

png("credit_score_distribution.png", width = 800, height = 600)
ggplot(data, aes(x = credit_score, fill = loan_approved)) +
  geom_histogram(bins = 15, alpha = 0.7, position = "identity") +
  labs(title = "Credit Score Distribution by Loan Approval",
       x = "Credit Score", y = "Frequency", fill = "Loan Approved") +
  theme_minimal()
dev.off()

cat("Data exploration completed. Check the generated plots:\n")
cat("- correlation_matrix.png\n")
cat("- age_income_plot.png\n")
cat("- credit_score_distribution.png\n")