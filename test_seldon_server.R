# Test Seldon Core Model Server with curl requests
# This script demonstrates how to make predictions using curl

library(jsonlite)

cat("🚀 Seldon Core Model Server Test Script\n")
cat("=======================================\n\n")

# Test data samples
test_samples <- list(
  # High probability approval
  high_approval = list(
    age = 35,
    income = 75000,
    education = 16,
    experience = 10,
    credit_score = 750
  ),
  
  # Low probability approval  
  low_approval = list(
    age = 22,
    income = 25000,
    education = 12,
    experience = 1,
    credit_score = 500
  ),
  
  # Medium probability
  medium_approval = list(
    age = 28,
    income = 45000,
    education = 14,
    experience = 5,
    credit_score = 650
  )
)

# Function to generate curl commands
generate_curl_commands <- function() {
  cat("📋 CURL Commands for Seldon Model Server:\n")
  cat("=========================================\n\n")
  
  # Health check
  cat("1️⃣ Health Check:\n")
  cat('curl -X GET "http://localhost:9090/health"\n\n')
  
  # Model metadata
  cat("2️⃣ Model Metadata:\n")
  cat('curl -X GET "http://localhost:9090/api/v1.0/metadata/loan-approval-model"\n\n')
  
  # Predictions
  cat("3️⃣ Make Predictions:\n\n")
  
  for (name in names(test_samples)) {
    sample_data <- test_samples[[name]]
    
    # Create Seldon-compatible JSON payload
    payload <- list(
      data = list(
        ndarray = list(list(
          sample_data$age,
          sample_data$income, 
          sample_data$education,
          sample_data$experience,
          sample_data$credit_score
        ))
      )
    )
    
    json_payload <- toJSON(payload, auto_unbox = TRUE, pretty = TRUE)
    
    cat("📊", toupper(name), "Sample:\n")
    cat("Data:", toJSON(sample_data, auto_unbox = TRUE), "\n\n")
    
    # Windows PowerShell curl command
    cat("PowerShell Command:\n")
    cat('$body = \'', gsub("'", "''", json_payload), '\'', "\n", sep="")
    cat('Invoke-RestMethod -Uri "http://localhost:9090/api/v1.0/predictions" -Method POST -Body $body -ContentType "application/json"', "\n\n")
    
    # Standard curl command
    cat("Standard curl:\n")
    escaped_json <- gsub('"', '\\"', json_payload)
    cat('curl -X POST "http://localhost:9090/api/v1.0/predictions" \\', "\n")
    cat('     -H "Content-Type: application/json" \\', "\n")  
    cat('     -d "', escaped_json, '"', "\n\n")
    
    cat("---\n\n")
  }
}

# Function to test the server (when running)
test_server <- function() {
  cat("🧪 Testing Seldon Server (if running):\n")
  cat("=====================================\n\n")
  
  tryCatch({
    # Test health endpoint
    response <- httr::GET("http://localhost:9090/health")
    if (response$status_code == 200) {
      cat("✅ Server is healthy!\n")
      
      # Test prediction
      sample_data <- test_samples$high_approval
      payload <- list(
        data = list(
          ndarray = list(list(
            sample_data$age,
            sample_data$income,
            sample_data$education, 
            sample_data$experience,
            sample_data$credit_score
          ))
        )
      )
      
      pred_response <- httr::POST(
        "http://localhost:9090/api/v1.0/predictions",
        body = payload,
        encode = "json"
      )
      
      if (pred_response$status_code == 200) {
        result <- httr::content(pred_response)
        cat("✅ Prediction successful!\n")
        cat("📊 Result:", toJSON(result, auto_unbox = TRUE, pretty = TRUE), "\n")
      } else {
        cat("❌ Prediction failed:", pred_response$status_code, "\n")
      }
      
    } else {
      cat("❌ Server not responding\n")
    }
  }, error = function(e) {
    cat("❌ Server not available:", e$message, "\n")
    cat("💡 Make sure to run: docker-compose -f docker-compose-seldon.yml up -d\n")
  })
}

# Main execution
cat("🎯 Deployment Instructions:\n")
cat("===========================\n")
cat("1. Build and start services:\n")
cat("   docker-compose -f docker-compose-seldon.yml up --build -d\n\n")
cat("2. Wait for services to be ready (check logs):\n")
cat("   docker-compose -f docker-compose-seldon.yml logs loan-model-server\n\n")
cat("3. Test the endpoints using the curl commands below\n\n")

# Generate curl commands
generate_curl_commands()

# Try to test if server is running
cat("🔍 Checking if server is currently running...\n")
test_server()