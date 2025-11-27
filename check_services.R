# MLflow Setup and Health Check Script
library(httr)
library(jsonlite)

# Configuration
MLFLOW_URL <- "http://localhost:5000"
MINIO_URL <- "http://localhost:9001"

# Function to check if service is running
check_service <- function(url, service_name) {
  tryCatch({
    response <- GET(url, timeout(10))
    if (status_code(response) %in% c(200, 401, 403)) {  # 401/403 might be expected for some endpoints
      cat("✅", service_name, "is running at", url, "\n")
      return(TRUE)
    } else {
      cat("❌", service_name, "returned status code:", status_code(response), "\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("❌", service_name, "is not accessible:", e$message, "\n")
    return(FALSE)
  })
}

# Function to check MLflow API
check_mlflow_api <- function() {
  url <- paste0(MLFLOW_URL, "/api/2.0/mlflow/experiments/list")
  
  tryCatch({
    response <- GET(url, timeout(10))
    if (status_code(response) == 200) {
      result <- fromJSON(content(response, "text"))
      cat("✅ MLflow API is working\n")
      
      if (length(result$experiments) > 0) {
        cat("📊 Found", length(result$experiments), "experiment(s):\n")
        for (exp in result$experiments) {
          cat("  -", exp$name, "(ID:", exp$experiment_id, ")\n")
        }
      } else {
        cat("📊 No experiments found yet\n")
      }
      return(TRUE)
    } else {
      cat("❌ MLflow API error. Status code:", status_code(response), "\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("❌ MLflow API error:", e$message, "\n")
    return(FALSE)
  })
}

cat("=== MLflow + MinIO Health Check ===\n")
cat("Checking services...\n\n")

# Check services
mlflow_ok <- check_service(MLFLOW_URL, "MLflow UI")
minio_ok <- check_service(MINIO_URL, "MinIO Console")

if (mlflow_ok) {
  check_mlflow_api()
}

cat("\n=== Setup Instructions ===\n")

if (!mlflow_ok || !minio_ok) {
  cat("🚀 To start services, run:\n")
  cat("   docker-compose up -d\n\n")
  cat("⏳ Wait for services to start, then run this script again\n\n")
  cat("📋 To check service status:\n")
  cat("   docker-compose ps\n\n")
  cat("📋 To view logs:\n")
  cat("   docker-compose logs mlflow\n")
  cat("   docker-compose logs minio\n\n")
} else {
  cat("✅ All services are running!\n\n")
  cat("🎯 Next steps:\n")
  cat("1. Run model training: Rscript train_model_mlflow.R\n")
  cat("2. Open MLflow UI: http://localhost:5000\n")
  cat("3. Open MinIO Console: http://localhost:9001\n")
  cat("   Login: minioadmin / minioadmin123\n\n")
}

cat("=== Service Information ===\n")
cat("MLflow UI:      http://localhost:5000\n")
cat("MinIO Console:  http://localhost:9001 (minioadmin/minioadmin123)\n")
cat("MinIO API:      http://localhost:9000\n")
cat("PostgreSQL:     localhost:5432 (mlflow/mlflow123)\n")