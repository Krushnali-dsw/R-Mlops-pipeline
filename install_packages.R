# Install Required Packages Script
# Run this script first to install all required packages

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# List of required packages
required_packages <- c("caret", "randomForest", "e1071", "corrplot", 
                      "ggplot2", "glmnet", "pROC")

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE)) {
      cat("Installing package:", package, "\n")
      install.packages(package, dependencies = TRUE)
      library(package, character.only = TRUE)
    } else {
      cat("Package", package, "is already installed\n")
    }
  }
}

# Install packages
install_if_missing(required_packages)

cat("\nAll required packages are now installed!\n")
cat("You can now run the other R scripts:\n")
cat("1. data_exploration.R - for exploratory data analysis\n")
cat("2. train_model.R - to train machine learning models\n")
cat("3. predict.R - to make predictions with the trained model\n")