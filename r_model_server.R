#!/usr/bin/env Rscript

# Pure R Microservice for Seldon Core
# Uses Plumber for REST API

library(randomForest)
library(jsonlite)
library(plumber)

# Global variables
model <- NULL
feature_names <- c("age", "income", "education", "experience", "credit_score")

# Load the trained model
load_model <- function() {
  model_path <- "/app/model/flask_random_forest.rds"
  if (file.exists(model_path)) {
    model <<- readRDS(model_path)
    cat("✅ Model loaded successfully from", model_path, "\n")
    return(TRUE)
  } else {
    cat("❌ Model file not found:", model_path, "\n")
    return(FALSE)
  }
}

# Initialize model on startup
if (!load_model()) {
  stop("Failed to load model")
}

#* @apiTitle Loan Approval Model API
#* @apiDescription R-based loan approval prediction microservice for Seldon Core

#* Health check endpoint
#* @get /health
function() {
  list(
    status = "healthy",
    model = "loan-approval-r-model",
    model_loaded = !is.null(model),
    timestamp = as.character(Sys.time()),
    version = "1.0.0"
  )
}

#* Get model metadata
#* @get /metadata
function() {
  list(
    name = "loan-approval-r-model",
    versions = list("v1.0.0"),
    platform = "R + Plumber",
    inputs = list(
      list(
        name = "age",
        datatype = "INT64",
        shape = list(1)
      ),
      list(
        name = "income", 
        datatype = "INT64",
        shape = list(1)
      ),
      list(
        name = "education",
        datatype = "INT64", 
        shape = list(1)
      ),
      list(
        name = "experience",
        datatype = "INT64",
        shape = list(1)
      ),
      list(
        name = "credit_score",
        datatype = "INT64",
        shape = list(1)
      )
    ),
    outputs = list(
      list(
        name = "prediction",
        datatype = "STRING",
        shape = list(1)
      ),
      list(
        name = "probability",
        datatype = "FP64",
        shape = list(1)
      )
    )
  )
}

#* Standard prediction endpoint
#* @post /predict
#* @param req The request object
function(req) {
  tryCatch({
    # Parse JSON body
    body <- jsonlite::fromJSON(rawToChar(req$postBody))
    
    # Handle different input formats
    if (is.null(body$instances) && is.null(body$data)) {
      # Direct format: {"age":35, "income":75000, ...}
      input_data <- body
    } else if (!is.null(body$instances)) {
      # Seldon format: {"instances": [{"age":35, "income":75000, ...}]}
      input_data <- body$instances[[1]]
    } else if (!is.null(body$data)) {
      # Array format: {"data": [35, 75000, 16, 10, 750]}
      if (is.list(body$data) && "ndarray" %in% names(body$data)) {
        # Seldon ndarray format
        input_data <- setNames(as.list(body$data$ndarray[[1]]), feature_names)
      } else {
        # Simple array format
        input_data <- setNames(as.list(body$data), feature_names)
      }
    } else {
      stop("Invalid input format")
    }
    
    # Validate input
    missing_features <- setdiff(feature_names, names(input_data))
    if (length(missing_features) > 0) {
      stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
    }
    
    # Create data frame for prediction
    pred_data <- data.frame(
      age = as.numeric(input_data$age),
      income = as.numeric(input_data$income),
      education = as.numeric(input_data$education),
      experience = as.numeric(input_data$experience),
      credit_score = as.numeric(input_data$credit_score)
    )
    
    # Make prediction
    prediction <- predict(model, pred_data, type = "class")
    probability <- predict(model, pred_data, type = "prob")
    
    # Get probability for approved class
    prob_approved <- if ("approved" %in% colnames(probability)) {
      probability[1, "approved"]
    } else {
      probability[1, 1]
    }
    
    # Format response
    result <- list(
      predictions = list(list(
        prediction = as.character(prediction),
        probability = as.numeric(prob_approved),
        confidence = as.numeric(abs(prob_approved - 0.5) * 2),
        input = input_data
      )),
      model_name = "loan-approval-r-model",
      model_version = "v1.0.0",
      timestamp = as.character(Sys.time())
    )
    
    cat("📊 Prediction made:", prediction, "with probability:", prob_approved, "\n")
    return(result)
    
  }, error = function(e) {
    cat("❌ Prediction error:", e$message, "\n")
    list(
      error = e$message,
      status = "error",
      timestamp = as.character(Sys.time())
    )
  })
}

#* Seldon Core compatible prediction endpoint
#* @post /api/v1.0/predictions
#* @param req The request object
function(req) {
  tryCatch({
    # Use the same prediction logic but format for Seldon Core
    result <- predict_handler(req)
    
    # Seldon Core expected format
    if (is.null(result$error)) {
      pred <- result$predictions[[1]]
      list(
        data = list(
          names = c("prediction", "probability"),
          ndarray = list(list(pred$prediction, pred$probability))
        ),
        meta = list(
          model_name = "loan-approval-r-model",
          model_version = "v1.0.0"
        )
      )
    } else {
      result
    }
  }, error = function(e) {
    list(
      error = e$message,
      status = "error"
    )
  })
}

# Helper function for prediction logic
predict_handler <- function(req) {
  # Parse JSON body
  body <- jsonlite::fromJSON(rawToChar(req$postBody))
  
  # Handle different input formats
  if (is.null(body$instances) && is.null(body$data)) {
    input_data <- body
  } else if (!is.null(body$instances)) {
    input_data <- body$instances[[1]]
  } else if (!is.null(body$data)) {
    if (is.list(body$data) && "ndarray" %in% names(body$data)) {
      input_data <- setNames(as.list(body$data$ndarray[[1]]), feature_names)
    } else {
      input_data <- setNames(as.list(body$data), feature_names)
    }
  } else {
    stop("Invalid input format")
  }
  
  # Validate input
  missing_features <- setdiff(feature_names, names(input_data))
  if (length(missing_features) > 0) {
    stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
  }
  
  # Create data frame for prediction
  pred_data <- data.frame(
    age = as.numeric(input_data$age),
    income = as.numeric(input_data$income),
    education = as.numeric(input_data$education),
    experience = as.numeric(input_data$experience),
    credit_score = as.numeric(input_data$credit_score)
  )
  
  # Make prediction
  prediction <- predict(model, pred_data, type = "class")
  probability <- predict(model, pred_data, type = "prob")
  
  # Get probability for approved class
  prob_approved <- if ("approved" %in% colnames(probability)) {
    probability[1, "approved"]
  } else {
    probability[1, 1]
  }
  
  # Format response
  list(
    predictions = list(list(
      prediction = as.character(prediction),
      probability = as.numeric(prob_approved),
      confidence = as.numeric(abs(prob_approved - 0.5) * 2),
      input = input_data
    )),
    model_name = "loan-approval-r-model",
    model_version = "v1.0.0"
  )
}

#* Root endpoint with API documentation
#* @get /
function() {
  list(
    message = "Loan Approval R Model Server",
    version = "1.0.0",
    endpoints = list(
      health = "GET /health - Health check",
      predict = "POST /predict - Make predictions",
      metadata = "GET /metadata - Model information",
      seldon_predictions = "POST /api/v1.0/predictions - Seldon Core predictions"
    ),
    example_request = list(
      age = 35,
      income = 75000,
      education = 16,
      experience = 10,
      credit_score = 750
    )
  )
}

# Start the server
cat("🚀 Starting R Model Server on port 9000...\n")
cat("📋 Model loaded with features:", paste(feature_names, collapse = ", "), "\n")
cat("🔗 Available endpoints:\n")
cat("   GET  /health - Health check\n")
cat("   POST /predict - Make predictions\n")  
cat("   GET  /metadata - Model metadata\n")
cat("   POST /api/v1.0/predictions - Seldon Core endpoint\n")
cat("   GET  / - API documentation\n")

# Create and run the API
pr <- plumb("/app/r_model_server.R")
pr$run(host = "0.0.0.0", port = 9000)