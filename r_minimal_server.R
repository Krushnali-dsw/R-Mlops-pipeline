# Minimal R Model Server
library(plumber)
library(randomForest)
library(jsonlite)

# Load model
model <- readRDS("flask_random_forest.rds")

#* Health check
#* @get /health
function() {
  list(status = "healthy", timestamp = Sys.time())
}

#* Predict endpoint
#* @post /predict
function(req) {
  # Parse JSON body
  body <- jsonlite::fromJSON(rawToChar(req$postBody))
  
  # Extract features
  input_data <- data.frame(
    age = body$age,
    income = body$income,
    education = body$education,
    experience = body$experience,
    credit_score = body$credit_score
  )
  
  # Predict
  prediction <- predict(model, input_data)
  prob <- predict(model, input_data, type = "prob")[,2]
  
  # Return result
  list(
    prediction = as.character(prediction),
    probability = round(prob, 4),
    timestamp = Sys.time()
  )
}