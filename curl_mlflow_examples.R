# MLflow REST API Examples using curl with data bodies
# This script demonstrates various MLflow operations using curl commands

library(jsonlite)

cat("🚀 MLflow REST API Examples with curl\n")
cat("====================================\n\n")

# Helper function to execute curl commands and parse responses
execute_curl <- function(description, command) {
  cat("📡", description, "\n")
  cat("Command:", command, "\n")
  
  # Execute the curl command
  result <- system(command, intern = TRUE)
  
  # Try to parse JSON response
  tryCatch({
    if (length(result) > 0) {
      json_response <- fromJSON(paste(result, collapse = ""))
      cat("Response:\n")
      print(json_response)
    }
  }, error = function(e) {
    cat("Raw Response:", paste(result, collapse = "\n"), "\n")
  })
  
  cat(strrep("-", 50), "\n\n")
  return(result)
}

# 1. Create a new experiment
cat("1️⃣ Creating New Experiment\n")
create_experiment_cmd <- 'curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/create" -H "Content-Type: application/json" -d "{\\"name\\": \\"curl_demo_experiment\\", \\"tags\\": [{\\"key\\": \\"environment\\", \\"value\\": \\"development\\"}, {\\"key\\": \\"created_by\\", \\"value\\": \\"curl_demo\\"}]}"'

execute_curl("Creating new experiment", create_experiment_cmd)

# 2. List all experiments
cat("2️⃣ Listing All Experiments\n")
list_experiments_cmd <- 'curl -X POST "http://localhost:5000/api/2.0/mlflow/experiments/search" -H "Content-Type: application/json" -d "{\\"max_results\\": 10}"'

experiments_result <- execute_curl("Listing experiments", list_experiments_cmd)

# 3. Create a new run
cat("3️⃣ Creating New MLflow Run\n")
# First, let's get an experiment ID from the previous result
exp_id <- "930346870948111632"  # Using existing experiment ID

create_run_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/create" -H "Content-Type: application/json" -d "{\\"experiment_id\\": \\"', exp_id, '\\", \\"run_name\\": \\"Curl Demo Run\\", \\"tags\\": [{\\"key\\": \\"mlflow.runName\\", \\"value\\": \\"Curl Demo Run\\"}, {\\"key\\": \\"model_type\\", \\"value\\": \\"demonstration\\"}]}"')

run_result <- execute_curl("Creating new run", create_run_cmd)

# Extract run ID for subsequent operations
run_id <- NULL
tryCatch({
  run_data <- fromJSON(paste(run_result, collapse = ""))
  run_id <- run_data$run$info$run_id
  cat("🆔 Created Run ID:", run_id, "\n\n")
}, error = function(e) {
  cat("⚠️ Could not extract run ID\n\n")
})

if (!is.null(run_id)) {
  # 4. Log parameters
  cat("4️⃣ Logging Parameters\n")
  log_param_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"key\\": \\"learning_rate\\", \\"value\\": \\"0.01\\"}"')
  
  execute_curl("Logging parameter: learning_rate", log_param_cmd)
  
  # Log multiple parameters
  log_param2_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-parameter" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"key\\": \\"batch_size\\", \\"value\\": \\"32\\"}"')
  
  execute_curl("Logging parameter: batch_size", log_param2_cmd)
  
  # 5. Log metrics
  cat("5️⃣ Logging Metrics\n")
  log_metric_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"key\\": \\"accuracy\\", \\"value\\": 0.95, \\"timestamp\\": ', as.numeric(Sys.time()) * 1000, ', \\"step\\": 1}"')
  
  execute_curl("Logging metric: accuracy", log_metric_cmd)
  
  # Log another metric
  log_metric2_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-metric" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"key\\": \\"loss\\", \\"value\\": 0.05, \\"timestamp\\": ', as.numeric(Sys.time()) * 1000, ', \\"step\\": 1}"')
  
  execute_curl("Logging metric: loss", log_metric2_cmd)
  
  # 6. Log batch of metrics
  cat("6️⃣ Logging Batch Metrics\n")
  batch_metrics <- list(
    list(key = "precision", value = 0.92, timestamp = as.numeric(Sys.time()) * 1000, step = 1),
    list(key = "recall", value = 0.89, timestamp = as.numeric(Sys.time()) * 1000, step = 1),
    list(key = "f1_score", value = 0.905, timestamp = as.numeric(Sys.time()) * 1000, step = 1)
  )
  
  batch_data <- list(run_id = run_id, metrics = batch_metrics)
  batch_json <- toJSON(batch_data, auto_unbox = TRUE)
  
  # Write JSON to temporary file for complex data
  temp_file <- "temp_batch_metrics.json"
  writeLines(batch_json, temp_file)
  
  batch_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/log-batch" -H "Content-Type: application/json" -d @', temp_file)
  
  execute_curl("Logging batch metrics", batch_cmd)
  
  # Clean up temp file
  if (file.exists(temp_file)) file.remove(temp_file)
  
  # 7. Set tags
  cat("7️⃣ Setting Tags\n")
  set_tag_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/set-tag" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"key\\": \\"model_version\\", \\"value\\": \\"v1.0\\"}"')
  
  execute_curl("Setting tag: model_version", set_tag_cmd)
  
  # 8. Update run status
  cat("8️⃣ Updating Run Status\n")
  update_run_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/update" -H "Content-Type: application/json" -d "{\\"run_id\\": \\"', run_id, '\\", \\"status\\": \\"FINISHED\\"}"')
  
  execute_curl("Updating run status to FINISHED", update_run_cmd)
  
  # 9. Get run details
  cat("9️⃣ Getting Run Details\n")
  get_run_cmd <- paste0('curl -X GET "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=', run_id, '"')
  
  execute_curl("Getting run details", get_run_cmd)
}

# 10. Search runs with complex query
cat("🔟 Searching Runs with Filter\n")
search_data <- list(
  experiment_ids = list(exp_id),
  filter = "metrics.accuracy > 0.9",
  max_results = 5,
  order_by = list("metrics.accuracy DESC")
)
search_json <- toJSON(search_data, auto_unbox = TRUE)

# Write to temp file for complex search
temp_search_file <- "temp_search.json"
writeLines(search_json, temp_search_file)

search_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/runs/search" -H "Content-Type: application/json" -d @', temp_search_file)

execute_curl("Searching runs with accuracy > 0.9", search_cmd)

# Clean up
if (file.exists(temp_search_file)) file.remove(temp_search_file)

# 11. Create model version (if model exists)
cat("1️⃣1️⃣ Model Registry Operations\n")
if (!is.null(run_id)) {
  # First create a registered model
  create_model_cmd <- 'curl -X POST "http://localhost:5000/api/2.0/mlflow/registered-models/create" -H "Content-Type: application/json" -d "{\\"name\\": \\"curl_demo_model\\", \\"description\\": \\"Model created via curl demo\\"}"'
  
  execute_curl("Creating registered model", create_model_cmd)
  
  # Create model version
  artifact_uri <- paste0("s3://mlflow-artifacts/", exp_id, "/", run_id, "/artifacts")
  create_version_cmd <- paste0('curl -X POST "http://localhost:5000/api/2.0/mlflow/model-versions/create" -H "Content-Type: application/json" -d "{\\"name\\": \\"curl_demo_model\\", \\"source\\": \\"', artifact_uri, '\\", \\"run_id\\": \\"', run_id, '\\"}"')
  
  execute_curl("Creating model version", create_version_cmd)
}

cat("\n🎉 MLflow curl demonstration complete!\n")
cat("📊 Check MLflow UI at: http://localhost:5000\n")
cat("🗂️ Check MinIO Console at: http://localhost:9001\n")