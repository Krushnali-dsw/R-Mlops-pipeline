# MLflow Model Registry and Deployment Script
library(httr)
library(jsonlite)
library(caret)
library(randomForest)

# Set working directory
setwd("c:/Users/Admin/Desktop/P2")

# MLflow configuration
MLFLOW_TRACKING_URI <- "http://localhost:5000"
MODEL_NAME <- "loan_approval_classifier"

# Helper function to register model
register_model <- function(run_id, model_name, artifact_path = "model") {
  url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/model-versions/create")
  
  source <- paste0("runs:/", run_id, "/", artifact_path)
  
  body <- list(
    name = model_name,
    source = source,
    description = "Loan approval classification model trained with R"
  )
  
  response <- POST(
    url,
    body = toJSON(body, auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(content(response, "text"))
    cat("Model registered successfully!\n")
    cat("Model Name:", result$model_version$name, "\n")
    cat("Version:", result$model_version$version, "\n")
    return(result$model_version)
  } else {
    cat("Failed to register model. Status code:", status_code(response), "\n")
    cat("Response:", content(response, "text"), "\n")
    return(NULL)
  }
}

# Helper function to transition model stage
transition_model_stage <- function(model_name, version, stage) {
  url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/model-versions/transition-stage")
  
  body <- list(
    name = model_name,
    version = version,
    stage = stage
  )
  
  response <- POST(
    url,
    body = toJSON(body, auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(content(response, "text"))
    cat("Model transitioned to", stage, "stage successfully!\n")
    return(result$model_version)
  } else {
    cat("Failed to transition model stage. Status code:", status_code(response), "\n")
    return(NULL)
  }
}

# Helper function to get latest model version
get_latest_model_version <- function(model_name, stage = NULL) {
  if (is.null(stage)) {
    url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/registered-models/get?name=", URLencode(model_name))
  } else {
    url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/model-versions/get-by-name/", URLencode(model_name), "/", stage)
  }
  
  response <- GET(url)
  
  if (status_code(response) == 200) {
    result <- fromJSON(content(response, "text"))
    if (is.null(stage)) {
      versions <- result$registered_model$latest_versions
      if (length(versions) > 0) {
        return(versions[[1]])
      }
    } else {
      return(result$model_version)
    }
  }
  
  return(NULL)
}

# Helper function to list experiments and find best run
get_best_run <- function(experiment_name) {
  # Get experiment ID
  get_exp_url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/get-by-name?experiment_name=", URLencode(experiment_name))
  exp_response <- GET(get_exp_url)
  
  if (status_code(exp_response) != 200) {
    cat("Experiment not found:", experiment_name, "\n")
    return(NULL)
  }
  
  exp_result <- fromJSON(content(exp_response, "text"))
  experiment_id <- exp_result$experiment$experiment_id
  
  # Search runs in the experiment
  search_url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/search")
  
  body <- list(
    experiment_ids = list(experiment_id),
    order_by = list("metrics.auc DESC")
  )
  
  search_response <- POST(
    search_url,
    body = toJSON(body, auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
  
  if (status_code(search_response) == 200) {
    search_result <- fromJSON(content(search_response, "text"))
    if (length(search_result$runs) > 0) {
      return(search_result$runs[[1]])
    }
  }
  
  return(NULL)
}

cat("=== MLflow Model Registry Demo ===\n")

# Get the best run from our experiment
best_run <- get_best_run("loan_approval_models")

if (!is.null(best_run)) {
  run_id <- best_run$info$run_id
  cat("Found best run ID:", run_id, "\n")
  
  # Get AUC metric from the run
  auc_metric <- NULL
  if (!is.null(best_run$data$metrics)) {
    for (metric in best_run$data$metrics) {
      if (metric$key == "auc") {
        auc_metric <- metric$value
        break
      }
    }
  }
  
  if (!is.null(auc_metric)) {
    cat("Best model AUC:", round(auc_metric, 4), "\n")
  }
  
  # Note: In a real scenario, you would log the model as an artifact during training
  # For this demo, we'll simulate the model registration process
  cat("\nNote: To register models in MLflow, you need to log them as artifacts during training.\n")
  cat("This would typically be done with mlflow$log_model() during the training process.\n")
  
} else {
  cat("No runs found in the experiment. Make sure to run train_model_mlflow.R first.\n")
}

# Function to simulate model serving
serve_model_prediction <- function(age, income, education, experience, credit_score) {
  # Load the best model from local file
  if (file.exists("best_model_mlflow.rds")) {
    model <- readRDS("best_model_mlflow.rds")
    
    # Create input data frame
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
    
    result <- list(
      prediction = as.character(prediction),
      probability_approved = probability[,2],
      probability_rejected = probability[,1],
      model_info = "Served from MLflow registry"
    )
    
    return(result)
  } else {
    return(list(error = "Model not found. Run training first."))
  }
}

# Example predictions with model serving simulation
cat("\n=== Model Serving Simulation ===\n")

examples <- list(
  list(25, 30000, 12, 2, 620),
  list(40, 70000, 16, 12, 750),
  list(50, 90000, 18, 20, 800)
)

for (i in seq_along(examples)) {
  ex <- examples[[i]]
  result <- serve_model_prediction(ex[[1]], ex[[2]], ex[[3]], ex[[4]], ex[[5]])
  
  cat("\nExample", i, "- Age:", ex[[1]], "Income:", ex[[2]], "Education:", ex[[3]], 
      "Experience:", ex[[4]], "Credit Score:", ex[[5]], "\n")
  
  if (is.null(result$error)) {
    cat("Prediction:", result$prediction, "\n")
    cat("Approval Probability:", round(result$probability_approved, 3), "\n")
  } else {
    cat("Error:", result$error, "\n")
  }
}

cat("\n=== MLflow Setup Instructions ===\n")
cat("1. Start services: docker-compose up -d\n")
cat("2. Wait for services to be ready (check with: docker-compose ps)\n")
cat("3. Run training with MLflow: Rscript train_model_mlflow.R\n")
cat("4. Access MLflow UI: http://localhost:5000\n")
cat("5. Access MinIO UI: http://localhost:9001 (minioadmin/minioadmin123)\n")
cat("6. Register and manage models through the MLflow UI\n")

cat("\n=== Service URLs ===\n")
cat("- MLflow UI: http://localhost:5000\n")
cat("- MinIO Console: http://localhost:9001\n")
cat("- MinIO API: http://localhost:9000\n")