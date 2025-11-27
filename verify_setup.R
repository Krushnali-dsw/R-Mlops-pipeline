# MLflow + MinIO Setup Verification
# This script verifies that our ML pipeline is working correctly

library(httr)
library(jsonlite)

cat("🔍 Verifying MLflow + MinIO Setup...\n\n")

# Check MLflow server health
check_mlflow <- function() {
  cat("1️⃣ Checking MLflow Server Health...\n")
  tryCatch({
    response <- GET("http://localhost:5000/health")
    if (response$status_code == 200) {
      cat("✅ MLflow server is running at http://localhost:5000\n")
      return(TRUE)
    }
  }, error = function(e) {
    cat("❌ MLflow server is not responding\n")
    return(FALSE)
  })
}

# Check MinIO server health
check_minio <- function() {
  cat("\n2️⃣ Checking MinIO Server Health...\n")
  tryCatch({
    response <- GET("http://localhost:9000/minio/health/live")
    if (response$status_code == 200) {
      cat("✅ MinIO server is running at http://localhost:9000\n")
      cat("🌐 MinIO Console available at http://localhost:9001\n")
      cat("   Username: minioadmin\n")
      cat("   Password: minioadmin123\n")
      return(TRUE)
    }
  }, error = function(e) {
    cat("❌ MinIO server is not responding\n")
    return(FALSE)
  })
}

# Check recent experiments
check_experiments <- function() {
  cat("\n3️⃣ Checking MLflow Experiments...\n")
  tryCatch({
    # Get experiments
    response <- POST(
      "http://localhost:5000/api/2.0/mlflow/experiments/search",
      body = list(max_results = 5),
      encode = "json"
    )
    
    if (response$status_code == 200) {
      experiments <- fromJSON(content(response, "text"))$experiments
      if (length(experiments) > 0) {
        cat("✅ Found", nrow(experiments), "experiment(s)\n")
        for (i in 1:nrow(experiments)) {
          cat("   📊", experiments$name[i], "(ID:", experiments$experiment_id[i], ")\n")
        }
        return(TRUE)
      }
    }
  }, error = function(e) {
    cat("❌ Could not retrieve experiments\n")
    return(FALSE)
  })
}

# Check recent runs
check_runs <- function() {
  cat("\n4️⃣ Checking Recent MLflow Runs...\n")
  tryCatch({
    # Get recent runs
    response <- POST(
      "http://localhost:5000/api/2.0/mlflow/runs/search",
      body = list(max_results = 3),
      encode = "json"
    )
    
    if (response$status_code == 200) {
      runs_data <- fromJSON(content(response, "text"))
      if (length(runs_data$runs) > 0) {
        runs <- runs_data$runs
        cat("✅ Found", length(runs), "recent run(s)\n")
        for (i in 1:length(runs)) {
          run <- runs[[i]]
          run_name <- ifelse(is.null(run$info$run_name), "Unnamed", run$info$run_name)
          cat("   🚀", run_name, "\n")
          cat("      Run ID:", run$info$run_id, "\n")
          cat("      Status:", run$info$status, "\n")
          cat("      Artifact URI:", run$info$artifact_uri, "\n")
          
          # Check if metrics exist
          if (length(run$data$metrics) > 0) {
            cat("      Metrics:\n")
            for (j in 1:length(run$data$metrics)) {
              metric <- run$data$metrics[[j]]
              cat("        📈", metric$key, ":", metric$value, "\n")
            }
          }
          cat("\n")
        }
        return(TRUE)
      }
    }
  }, error = function(e) {
    cat("❌ Could not retrieve runs\n")
    return(FALSE)
  })
}

# Run all checks
main <- function() {
  cat(strrep("=", 50), "\n")
  cat("🔬 MLflow + MinIO Setup Verification\n")
  cat(strrep("=", 50), "\n")
  
  mlflow_ok <- check_mlflow()
  minio_ok <- check_minio()
  exp_ok <- check_experiments()
  runs_ok <- check_runs()
  if(is.null(runs_ok)) runs_ok <- FALSE
  
  cat("\n\n")
  cat("📋 SUMMARY:\n")
  cat(strrep("=", 30), "\n")
  cat("MLflow Server: ", if(mlflow_ok) "✅ OK" else "❌ FAILED", "\n")
  cat("MinIO Server:  ", if(minio_ok) "✅ OK" else "❌ FAILED", "\n")
  cat("Experiments:   ", if(exp_ok) "✅ OK" else "❌ FAILED", "\n")
  cat("Recent Runs:   ", if(runs_ok) "✅ OK" else "❌ FAILED", "\n")
  
  if (mlflow_ok && minio_ok && exp_ok && runs_ok) {
    cat("\n🎉 All systems operational! Your ML pipeline is ready.\n")
    cat("\n📱 Access Points:\n")
    cat("   MLflow UI: http://localhost:5000\n")
    cat("   MinIO Console: http://localhost:9001\n")
  } else {
    cat("\n⚠️  Some issues detected. Please check the logs above.\n")
  }
}

# Execute verification
main()