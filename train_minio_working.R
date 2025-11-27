# Working MLflow Model Training with MinIO Artifacts
library(caret)
library(randomForest)
library(httr)
library(jsonlite)

# Set environment variables for S3 connection
Sys.setenv(
  MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000",
  AWS_ACCESS_KEY_ID = "minioadmin",
  AWS_SECRET_ACCESS_KEY = "minioadmin123"
)

# MLflow configuration
MLFLOW_TRACKING_URI <- "http://localhost:5000"
EXPERIMENT_NAME <- "loan_models_with_minio"

# Helper function to upload file via curl (most reliable method)
upload_artifact_via_curl <- function(run_id, file_path, artifact_path = "") {
  tryCatch({
    # Construct curl command
    if (artifact_path == "") {
      curl_cmd <- sprintf(
        'curl -X POST "%s/api/2.0/mlflow/artifacts" -H "Content-Type: multipart/form-data" -F "run_id=%s" -F "file=@%s"',
        MLFLOW_TRACKING_URI, run_id, file_path
      )
    } else {
      curl_cmd <- sprintf(
        'curl -X POST "%s/api/2.0/mlflow/artifacts" -H "Content-Type: multipart/form-data" -F "run_id=%s" -F "path=%s" -F "file=@%s"',
        MLFLOW_TRACKING_URI, run_id, artifact_path, file_path
      )
    }
    
    cat("📤 Uploading", basename(file_path), "...\n")
    result <- system(curl_cmd, intern = TRUE)
    cat("✅ Upload successful\n")
    return(TRUE)
  }, error = function(e) {
    cat("❌ Upload failed:", e$message, "\n")
    return(FALSE)
  })
}

# Start MLflow run
start_run <- function(experiment_name, run_name) {
  # Create/get experiment
  exp_response <- POST(
    paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/create"),
    body = toJSON(list(name = experiment_name), auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
  
  if (status_code(exp_response) == 400) {
    get_response <- GET(paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/get-by-name?experiment_name=", URLencode(experiment_name)))
    experiment_id <- fromJSON(content(get_response, "text"))$experiment$experiment_id
  } else {
    experiment_id <- fromJSON(content(exp_response, "text"))$experiment_id
  }
  
  # Start run
  run_response <- POST(
    paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/create"),
    body = toJSON(list(
      experiment_id = experiment_id,
      tags = list(list(key = "mlflow.runName", value = run_name))
    ), auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
  
  return(fromJSON(content(run_response, "text"))$run$info$run_id)
}

# Log parameter
log_param <- function(run_id, key, value) {
  POST(
    paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/log-parameter"),
    body = toJSON(list(run_id = run_id, key = key, value = as.character(value)), auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
}

# Log metric
log_metric <- function(run_id, key, value) {
  POST(
    paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/log-metric"),
    body = toJSON(list(
      run_id = run_id, 
      key = key, 
      value = as.numeric(value),
      timestamp = as.numeric(Sys.time()) * 1000
    ), auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
}

# End run
end_run <- function(run_id) {
  POST(
    paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/update"),
    body = toJSON(list(run_id = run_id, status = "FINISHED"), auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "raw"
  )
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

# Train Random Forest with MLflow tracking
cat("\n🚀 Training Random Forest with MLflow + MinIO...\n")

run_id <- start_run(EXPERIMENT_NAME, "Random Forest - MinIO Storage")
cat("✅ Started run:", run_id, "\n")

# Log parameters
log_param(run_id, "algorithm", "Random Forest")
log_param(run_id, "cv_folds", "5")
log_param(run_id, "features", paste(features, collapse = ","))

# Train model
cat("🔄 Training Random Forest...\n")
model <- train(x = X_train, y = y_train, method = "rf", trControl = ctrl, metric = "ROC", tuneLength = 3)

# Make predictions and calculate metrics
pred <- predict(model, X_test)
prob <- predict(model, X_test, type = "prob")[,2]
cm <- confusionMatrix(pred, y_test)
accuracy <- cm$overall['Accuracy']

roc_obj <- pROC::roc(y_test, prob, quiet = TRUE)
auc_value <- pROC::auc(roc_obj)

# Log metrics
log_metric(run_id, "accuracy", accuracy)
log_metric(run_id, "auc", auc_value)
log_metric(run_id, "sensitivity", cm$byClass['Sensitivity'])
log_metric(run_id, "specificity", cm$byClass['Specificity'])

# Log best parameters
if (!is.null(model$bestTune)) {
  for (param in names(model$bestTune)) {
    log_param(run_id, paste0("best_", param), model$bestTune[[param]])
  }
}

# Save and upload model
model_file <- "random_forest_minio.rds"
saveRDS(model, model_file)
upload_artifact_via_curl(run_id, model_file, "model")

# Create and upload model summary
summary_data <- list(
  algorithm = "Random Forest",
  accuracy = round(accuracy, 4),
  auc = round(auc_value, 4),
  best_params = model$bestTune,
  features = features,
  created_at = as.character(Sys.time())
)

summary_file <- "model_summary.json"
writeLines(toJSON(summary_data, pretty = TRUE, auto_unbox = TRUE), summary_file)
upload_artifact_via_curl(run_id, summary_file, "metadata")

# Create and upload feature importance plot
if (file.exists("feature_importance.png")) file.remove("feature_importance.png")
png("feature_importance.png", width = 800, height = 600)
importance_plot <- varImp(model)
plot(importance_plot, main = "Feature Importance - Random Forest")
dev.off()

upload_artifact_via_curl(run_id, "feature_importance.png", "plots")

# End the run
end_run(run_id)

cat("\n🎯 Training Complete!\n")
cat("📊 MLflow UI: http://localhost:5000\n") 
cat("🗄️  MinIO Console: http://localhost:9001\n")
cat("📈 Accuracy:", round(accuracy, 4), "| AUC:", round(auc_value, 4), "\n")

# Check MinIO contents
cat("\n🔍 Checking MinIO artifacts...\n")
Sys.sleep(3) # Give time for upload

tryCatch({
  system('docker run --rm --network p2_default --entrypoint="" minio/mc:latest sh -c "mc alias set minio http://mlflow-minio:9000 minioadmin minioadmin123 && echo \'=== MinIO Bucket Contents ===' && mc ls minio/mlflow-artifacts/ --recursive"', intern = FALSE)
}, error = function(e) {
  cat("Could not check MinIO contents automatically\n")
})