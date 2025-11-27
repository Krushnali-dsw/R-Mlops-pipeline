# Enhanced MLflow Model Training with Proper Artifact Logging
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
EXPERIMENT_NAME <- "loan_approval_models_with_artifacts"

# Helper function to log model artifacts to MLflow
log_model_artifact <- function(run_id, model, model_path, artifact_path = "model") {
  if (is.null(run_id)) return(invisible(NULL))
  
  tryCatch({
    # Save model locally first
    saveRDS(model, model_path)
    
    # Read the model file as binary
    model_content <- readBin(model_path, "raw", file.info(model_path)$size)
    model_b64 <- base64encode(model_content)
    
    # Upload to MLflow as artifact
    url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/artifacts")
    
    # Create form data
    boundary <- paste0("----formdata-", as.integer(Sys.time()), sample(1000:9999, 1))
    
    body_parts <- c(
      paste0("--", boundary),
      'Content-Disposition: form-data; name="run_id"',
      "",
      run_id,
      paste0("--", boundary),
      'Content-Disposition: form-data; name="path"',
      "",
      artifact_path,
      paste0("--", boundary),
      paste0('Content-Disposition: form-data; name="file"; filename="model.rds"'),
      "Content-Type: application/octet-stream",
      "",
      model_b64,
      paste0("--", boundary, "--")
    )
    
    body <- paste(body_parts, collapse = "\r\n")
    
    response <- POST(
      url,
      body = body,
      add_headers(
        "Content-Type" = paste0("multipart/form-data; boundary=", boundary)
      ),
      encode = "raw"
    )
    
    if (status_code(response) %in% c(200, 201)) {
      cat("📁 Model artifact uploaded to MLflow\n")
    } else {
      cat("⚠️  Failed to upload artifact. Status:", status_code(response), "\n")
      cat("Response:", content(response, "text"), "\n")
    }
  }, error = function(e) {
    cat("⚠️  Error uploading artifact:", e$message, "\n")
  })
}

# Enhanced start MLflow run function
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
    
    # Get experiment ID
    if (status_code(exp_response) %in% c(200, 400)) {
      if (status_code(exp_response) == 400) {
        get_response <- GET(paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/get-by-name?experiment_name=", URLencode(experiment_name)))
        experiment_id <- fromJSON(content(get_response, "text"))$experiment$experiment_id
      } else {
        experiment_id <- fromJSON(content(exp_response, "text"))$experiment_id
      }
      
      # Start run with artifact configuration
      run_body <- list(
        experiment_id = experiment_id,
        tags = list(
          list(key = "mlflow.runName", value = run_name %||% paste("Run", Sys.time())),
          list(key = "mlflow.source.type", value = "LOCAL"),
          list(key = "mlflow.user", value = "r_user")
        )
      )
      
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

# Helper functions (same as before)
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

# Load and prepare data
cat("📊 Loading dataset...\n")
data <- read.csv("dataset.csv")
data$loan_approved <- factor(data$loan_approved, levels = c(0, 1), labels = c("Rejected", "Approved"))

set.seed(123)
train_index <- createDataPartition(data$loan_approved, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

features <- c("age", "income", "education", "experience", "credit_score")
X_train <- train_data[, features]
y_train <- train_data$loan_approved
X_test <- test_data[, features]
y_test <- test_data$loan_approved

ctrl <- trainControl(method = "cv", number = 5, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = FALSE)

cat("\n🚀 Training models with MLflow tracking and artifact storage...\n")

# Enhanced training function with artifact logging
train_and_track_model_with_artifacts <- function(algorithm_name, method) {
  run_id <- start_mlflow_run(EXPERIMENT_NAME, paste(algorithm_name, "Run - With Artifacts"))
  
  if (!is.null(run_id)) {
    # Log parameters
    log_mlflow_param(run_id, "algorithm", algorithm_name)
    log_mlflow_param(run_id, "cv_folds", 5)
    log_mlflow_param(run_id, "train_size", nrow(train_data))
    log_mlflow_param(run_id, "test_size", nrow(test_data))
    log_mlflow_param(run_id, "features", paste(features, collapse = ","))
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
  
  auc_value <- tryCatch({
    roc_obj <- pROC::roc(y_test, prob, quiet = TRUE)
    pROC::auc(roc_obj)
  }, error = function(e) {
    warning("Could not calculate AUC:", e$message)
    NA
  })
  
  # Log metrics
  if (!is.null(run_id)) {
    log_mlflow_metric(run_id, "accuracy", accuracy)
    if (!is.na(auc_value)) {
      log_mlflow_metric(run_id, "auc", auc_value)
    }
    log_mlflow_metric(run_id, "sensitivity", cm$byClass['Sensitivity'])
    log_mlflow_metric(run_id, "specificity", cm$byClass['Specificity'])
    
    # Log best tuning parameters
    if (!is.null(model$bestTune)) {
      for (param_name in names(model$bestTune)) {
        log_mlflow_param(run_id, paste0("best_", param_name), model$bestTune[[param_name]])
      }
    }
  }
  
  # Save model and log as artifact
  model_filename <- paste0(tolower(gsub(" ", "_", algorithm_name)), "_with_artifacts.rds")
  log_model_artifact(run_id, model, model_filename, "model")
  
  # Also save model info as JSON
  model_info <- list(
    algorithm = algorithm_name,
    accuracy = accuracy,
    auc = if(!is.na(auc_value)) auc_value else NULL,
    train_time = Sys.time(),
    features = features,
    model_params = if(!is.null(model$bestTune)) model$bestTune else NULL
  )
  
  info_filename <- paste0(tolower(gsub(" ", "_", algorithm_name)), "_info.json")
  writeLines(toJSON(model_info, pretty = TRUE, auto_unbox = TRUE), info_filename)
  
  # Log model info as artifact
  if (!is.null(run_id)) {
    tryCatch({
      info_content <- readBin(info_filename, "raw", file.info(info_filename)$size)
      info_b64 <- base64encode(info_content)
      
      # Simple artifact upload for JSON
      url <- paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/artifacts")
      boundary <- paste0("----formdata-", as.integer(Sys.time()), sample(1000:9999, 1))
      
      body_parts <- c(
        paste0("--", boundary),
        'Content-Disposition: form-data; name="run_id"',
        "",
        run_id,
        paste0("--", boundary),
        'Content-Disposition: form-data; name="path"',
        "",
        "model_info",
        paste0("--", boundary),
        paste0('Content-Disposition: form-data; name="file"; filename="model_info.json"'),
        "Content-Type: application/json",
        "",
        info_b64,
        paste0("--", boundary, "--")
      )
      
      body <- paste(body_parts, collapse = "\r\n")
      
      response <- POST(
        url,
        body = body,
        add_headers("Content-Type" = paste0("multipart/form-data; boundary=", boundary)),
        encode = "raw"
      )
      
      if (status_code(response) %in% c(200, 201)) {
        cat("📄 Model info uploaded to MLflow\n")
      }
    }, error = function(e) {
      cat("⚠️  Could not upload model info:", e$message, "\n")
    })
  }
  
  # End run
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

# Train models with artifact logging
results <- list()

cat("\n=== Training with Artifact Storage ===\n")

# Train Random Forest
results$rf <- train_and_track_model_with_artifacts("Random Forest", "rf")

# Train Logistic Regression  
results$logistic <- train_and_track_model_with_artifacts("Logistic Regression", "glm")

# Find best model
aucs <- sapply(results, function(x) x$auc)
best_idx <- which.max(aucs)
best_model <- results[[best_idx]]
best_name <- names(results)[best_idx]

cat("\n🏆 Best Model Results:\n")
cat("Algorithm:", best_name, "\n")
cat("Accuracy:", round(best_model$accuracy, 4), "\n")
cat("AUC:", round(best_model$auc, 4), "\n")

# Copy best model
file.copy(best_model$filename, "best_model_with_artifacts.rds", overwrite = TRUE)

cat("\n🎯 Training with Artifacts Complete!\n")
cat("📊 MLflow UI: http://localhost:5000\n")
cat("🗄️  MinIO Console: http://localhost:9001\n")
cat("📁 Check MinIO bucket for uploaded artifacts!\n")

# Verify artifacts in MinIO
cat("\n🔍 Checking MinIO for artifacts...\n")
Sys.sleep(2) # Give time for artifacts to be uploaded