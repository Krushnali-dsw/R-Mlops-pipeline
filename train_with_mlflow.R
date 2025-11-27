# Enhanced R Model Training with MLflow Registration
library(caret)
library(randomForest)
library(e1071)
library(httr)
library(jsonlite)
library(base64enc)

# Set working directory
setwd("c:/Users/Admin/Desktop/P2")

# MLflow configuration
MLFLOW_TRACKING_URI <- "http://localhost:5000"
EXPERIMENT_NAME <- "loan_approval_models"

# Helper function to start MLflow run
start_mlflow_run <- function(experiment_name, run_name = NULL) {
  tryCatch({
    # Create experiment if it doesn't exist
    exp_body <- list(name = experiment_name)
    exp_response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/create"),
      body = toJSON(exp_body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    # Get experiment ID (either from creation or existing)
    if (status_code(exp_response) %in% c(200, 400)) {
      if (status_code(exp_response) == 400) {
        # Experiment exists, get its ID
        get_response <- GET(paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/get-by-name?experiment_name=", URLencode(experiment_name)))
        experiment_id <- fromJSON(content(get_response, "text"))$experiment$experiment_id
      } else {
        experiment_id <- fromJSON(content(exp_response, "text"))$experiment_id
      }
      
      # Start run
      run_body <- list(experiment_id = experiment_id)
      if (!is.null(run_name)) {
        run_body$tags <- list(list(key = "mlflow.runName", value = run_name))
      }
      
      run_response <- POST(
        paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/create"),
        body = toJSON(run_body, auto_unbox = TRUE),
        add_headers("Content-Type" = "application/json"),
        encode = "raw"
      )
      
      if (status_code(run_response) == 200) {
        run_info <- fromJSON(content(run_response, "text"))$run$info
        cat("✅ Started MLflow run:", run_info$run_id, "\n")
        return(run_info$run_id)
      }
    }
    
    warning("Failed to start MLflow run")
    return(NULL)
  }, error = function(e) {
    warning("MLflow connection failed:", e$message)
    return(NULL)
  })
}

# Helper function to log metrics
log_mlflow_metric <- function(run_id, key, value) {
  if (is.null(run_id)) return(invisible(NULL))
  
  tryCatch({
    body <- list(
      run_id = run_id,
      key = key,
      value = as.numeric(value),
      timestamp = as.numeric(Sys.time()) * 1000
    )
    
    response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/log-metric"),
      body = toJSON(body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    if (status_code(response) == 200) {
      cat("📊 Logged metric:", key, "=", value, "\n")
    }
  }, error = function(e) {
    warning("Failed to log metric:", e$message)
  })
}

# Helper function to log parameters
log_mlflow_param <- function(run_id, key, value) {
  if (is.null(run_id)) return(invisible(NULL))
  
  tryCatch({
    body <- list(
      run_id = run_id,
      key = key,
      value = as.character(value)
    )
    
    response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/log-parameter"),
      body = toJSON(body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    if (status_code(response) == 200) {
      cat("🔧 Logged parameter:", key, "=", value, "\n")
    }
  }, error = function(e) {
    warning("Failed to log parameter:", e$message)
  })
}

# Helper function to register model
register_model <- function(run_id, model_name, model_stage = "Staging") {
  if (is.null(run_id)) return(invisible(NULL))
  
  tryCatch({
    # First, create the registered model if it doesn't exist
    reg_body <- list(name = model_name)
    reg_response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/registered-models/create"),
      body = toJSON(reg_body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    # Create model version
    version_body <- list(
      name = model_name,
      source = paste0("runs:/", run_id, "/model"),
      description = paste("Loan approval model created on", Sys.Date())
    )
    
    version_response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/model-versions/create"),
      body = toJSON(version_body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    if (status_code(version_response) == 200) {
      version_info <- fromJSON(content(version_response, "text"))$model_version
      cat("🏷️  Registered model:", model_name, "version", version_info$version, "\n")
      
      # Transition to staging
      transition_body <- list(
        name = model_name,
        version = version_info$version,
        stage = model_stage
      )
      
      transition_response <- POST(
        paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/model-versions/transition-stage"),
        body = toJSON(transition_body, auto_unbox = TRUE),
        add_headers("Content-Type" = "application/json"),
        encode = "raw"
      )
      
      if (status_code(transition_response) == 200) {
        cat("🎯 Model promoted to", model_stage, "stage\n")
      }
      
      return(version_info$version)
    }
  }, error = function(e) {
    warning("Failed to register model:", e$message)
  })
  
  return(NULL)
}

# Helper function to end run
end_mlflow_run <- function(run_id, status = "FINISHED") {
  if (is.null(run_id)) return(invisible(NULL))
  
  tryCatch({
    body <- list(run_id = run_id, status = status)
    response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/update"),
      body = toJSON(body, auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    if (status_code(response) == 200) {
      cat("✅ MLflow run completed\n")
    }
  }, error = function(e) {
    warning("Failed to end run:", e$message)
  })
}

# Load and prepare the data
cat("📊 Loading dataset...\n")
data <- read.csv("dataset.csv")
data$loan_approved <- factor(data$loan_approved, levels = c(0, 1), labels = c("Rejected", "Approved"))

# Set seed for reproducibility
set.seed(123)

# Split data
train_index <- createDataPartition(data$loan_approved, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Prepare features
features <- c("age", "income", "education", "experience", "credit_score")
X_train <- train_data[, features]
y_train <- train_data$loan_approved
X_test <- test_data[, features]
y_test <- test_data$loan_approved

# Cross-validation control
ctrl <- trainControl(method = "cv", number = 5, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = FALSE)

cat("\n🚀 Training models with MLflow tracking...\n")

# Function to train and track model
train_and_track_model <- function(algorithm_name, method, model_name_suffix = "") {
  run_id <- start_mlflow_run(EXPERIMENT_NAME, paste(algorithm_name, "Run"))
  
  if (!is.null(run_id)) {
    # Log parameters
    log_mlflow_param(run_id, "algorithm", algorithm_name)
    log_mlflow_param(run_id, "cv_folds", 5)
    log_mlflow_param(run_id, "train_size", nrow(train_data))
    log_mlflow_param(run_id, "test_size", nrow(test_data))
  }
  
  # Train model
  cat("\n🔄 Training", algorithm_name, "...\n")
  
  if (method == "glm") {
    model <- train(x = X_train, y = y_train,
                   method = method, family = "binomial",
                   trControl = ctrl, metric = "ROC")
  } else {
    model <- train(x = X_train, y = y_train,
                   method = method,
                   trControl = ctrl, metric = "ROC",
                   tuneLength = 3)
  }
  
  # Make predictions
  pred <- predict(model, X_test)
  prob <- predict(model, X_test, type = "prob")[,2]
  
  # Calculate metrics
  cm <- confusionMatrix(pred, y_test)
  accuracy <- cm$overall['Accuracy']
  
  # Calculate AUC safely
  auc_value <- tryCatch({
    roc_obj <- pROC::roc(y_test, prob, quiet = TRUE)
    pROC::auc(roc_obj)
  }, error = function(e) {
    warning("Could not calculate AUC:", e$message)
    NA
  })
  
  # Log metrics to MLflow
  if (!is.null(run_id)) {
    log_mlflow_metric(run_id, "accuracy", accuracy)
    if (!is.na(auc_value)) {
      log_mlflow_metric(run_id, "auc", auc_value)
    }
    log_mlflow_metric(run_id, "sensitivity", cm$byClass['Sensitivity'])
    log_mlflow_metric(run_id, "specificity", cm$byClass['Specificity'])
  }
  
  # Save model locally
  model_filename <- paste0(tolower(gsub(" ", "_", algorithm_name)), "_model.rds")
  saveRDS(model, model_filename)
  
  # Register model in MLflow
  model_name <- paste0("loan_classifier_", tolower(gsub(" ", "_", algorithm_name)))
  if (!is.null(run_id) && !is.na(auc_value) && auc_value > 0.7) {
    register_model(run_id, model_name, "Staging")
  }
  
  # End MLflow run
  if (!is.null(run_id)) {
    end_mlflow_run(run_id)
  }
  
  cat("✅", algorithm_name, "completed - Accuracy:", round(accuracy, 4), 
      if(!is.na(auc_value)) paste("AUC:", round(auc_value, 4)) else "", "\n")
  
  return(list(
    model = model,
    accuracy = accuracy,
    auc = if(!is.na(auc_value)) auc_value else 0,
    filename = model_filename,
    run_id = run_id
  ))
}

# Train models
results <- list()

# 1. Random Forest
results$rf <- train_and_track_model("Random Forest", "rf")

# 2. Logistic Regression  
results$logistic <- train_and_track_model("Logistic Regression", "glm")

# 3. K-Nearest Neighbors
results$knn <- train_and_track_model("K-Nearest Neighbors", "knn")

# Find and register best model
aucs <- sapply(results, function(x) x$auc)
best_idx <- which.max(aucs)
best_model <- results[[best_idx]]
best_name <- names(results)[best_idx]

cat("\n🏆 Best Model Results:\n")
cat("Algorithm:", best_name, "\n")
cat("Accuracy:", round(best_model$accuracy, 4), "\n")
cat("AUC:", round(best_model$auc, 4), "\n")

# Copy best model
file.copy(best_model$filename, "best_model.rds", overwrite = TRUE)

# Register best model as production
if (!is.null(best_model$run_id)) {
  register_model(best_model$run_id, "loan_classifier_production", "Production")
}

cat("\n🎯 Training Complete!\n")
cat("📊 MLflow UI: http://localhost:5000\n")
cat("🗄️  MinIO Console: http://localhost:9001\n")
cat("📁 Best model saved as: best_model.rds\n")

# Test the best model
cat("\n🧪 Testing predictions:\n")
test_cases <- data.frame(
  age = c(25, 40, 50),
  income = c(30000, 70000, 90000),
  education = c(12, 16, 18),
  experience = c(2, 12, 20),
  credit_score = c(620, 750, 800)
)

predictions <- predict(best_model$model, test_cases)
probabilities <- predict(best_model$model, test_cases, type = "prob")

for (i in 1:nrow(test_cases)) {
  cat("Case", i, ":", as.character(predictions[i]), 
      "- Prob:", round(probabilities[i, 2], 3), "\n")
}