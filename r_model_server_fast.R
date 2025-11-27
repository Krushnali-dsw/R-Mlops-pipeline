#!/usr/bin/env Rscript

# Lightweight R Model Server
suppressMessages({
  library(randomForest)
  library(jsonlite)
  library(plumber)
})

# Load model
model <- readRDS("/app/model/flask_random_forest.rds")
cat("✅ Model loaded\n")

#* @apiTitle Fast Loan Model
#* @get /health
function() list(status = "ok")

#* @post /predict
function(req) {
  body <- fromJSON(rawToChar(req$postBody))
  
  # Simple prediction
  pred_data <- data.frame(
    age = body$age,
    income = body$income, 
    education = body$education,
    experience = body$experience,
    credit_score = body$credit_score
  )
  
  pred <- predict(model, pred_data, type = "prob")
  prob <- pred[,"approved"]
  
  list(
    prediction = ifelse(prob > 0.5, "approved", "denied"),
    probability = prob,
    confidence = abs(prob - 0.5) * 2
  )
}

# Start server
pr <- plumb("/app/r_model_server_fast.R")
pr$run(host = "0.0.0.0", port = 9000)