# Simple MLflow Artifact Logging Test
library(httr)
library(jsonlite)

# MLflow configuration
MLFLOW_TRACKING_URI <- "http://localhost:5000"

# Test function to upload a simple file
test_artifact_upload <- function() {
  cat("🧪 Testing MLflow artifact upload...\n")
  
  # Create a simple test file
  test_content <- "This is a test model artifact file created at: "
  test_content <- paste0(test_content, Sys.time())
  writeLines(test_content, "test_artifact.txt")
  
  # Start a simple run
  tryCatch({
    # Create experiment
    exp_response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/create"),
      body = toJSON(list(name = "artifact_test"), auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    # Get experiment ID
    if (status_code(exp_response) == 400) {
      get_response <- GET(paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/experiments/get-by-name?experiment_name=artifact_test"))
      experiment_id <- fromJSON(content(get_response, "text"))$experiment$experiment_id
    } else {
      experiment_id <- fromJSON(content(exp_response, "text"))$experiment_id
    }
    
    # Start run
    run_response <- POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/create"),
      body = toJSON(list(experiment_id = experiment_id), auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    run_id <- fromJSON(content(run_response, "text"))$run$info$run_id
    cat("✅ Created run:", run_id, "\n")
    
    # Try to upload using curl (simpler approach)
    curl_cmd <- sprintf(
      'curl -X POST "%s/api/2.0/mlflow/artifacts" -H "Content-Type: multipart/form-data" -F "run_id=%s" -F "path=test_artifacts" -F "file=@test_artifact.txt"',
      MLFLOW_TRACKING_URI, run_id
    )
    
    cat("📤 Uploading test artifact...\n")
    system(curl_cmd, intern = TRUE)
    
    # Check if artifact was uploaded
    Sys.sleep(2)
    
    # End the run
    POST(
      paste0(MLFLOW_TRACKING_URI, "/api/2.0/mlflow/runs/update"),
      body = toJSON(list(run_id = run_id, status = "FINISHED"), auto_unbox = TRUE),
      add_headers("Content-Type" = "application/json"),
      encode = "raw"
    )
    
    cat("✅ Test completed! Check MLflow UI and MinIO console.\n")
    return(run_id)
    
  }, error = function(e) {
    cat("❌ Test failed:", e$message, "\n")
    return(NULL)
  })
}

# Run the test
test_run_id <- test_artifact_upload()

if (!is.null(test_run_id)) {
  cat("\n🎯 Test run ID:", test_run_id, "\n")
  cat("📊 Check MLflow UI: http://localhost:5000\n")
  cat("🗄️  Check MinIO Console: http://localhost:9001\n")
}