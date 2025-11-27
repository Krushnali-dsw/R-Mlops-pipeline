#!/usr/bin/env python3
"""
Simple Flask Model Server for R Random Forest Model
Direct REST API without Seldon Core complexity
"""

from flask import Flask, request, jsonify
import os
import json
import subprocess
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model_ready = False
feature_names = ["age", "income", "education", "experience", "credit_score"]

# R prediction script template
r_prediction_script = '''
# Load required libraries
library(randomForest)
library(jsonlite)

# Load the model
model <- readRDS("/app/model/flask_random_forest.rds")

# Function to make predictions
predict_loan <- function(input_data) {
    # Convert input to data frame
    df <- data.frame(
        age = input_data$age,
        income = input_data$income,
        education = input_data$education,
        experience = input_data$experience,
        credit_score = input_data$credit_score,
        stringsAsFactors = FALSE
    )
    
    # Make prediction
    prediction <- predict(model, df, type = "prob")
    prob_approved <- prediction[,"approved"]
    
    # Return results
    result <- list(
        probability = as.numeric(prob_approved),
        prediction = ifelse(prob_approved > 0.5, "approved", "denied"),
        confidence = as.numeric(abs(prob_approved - 0.5) * 2)
    )
    
    return(result)
}

# Read input from stdin and make prediction
input_json <- readLines("stdin", warn = FALSE)
input_data <- fromJSON(input_json)
result <- predict_loan(input_data)
cat(toJSON(result, auto_unbox = TRUE))
'''

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global model_ready
    try:
        # Check if model file exists
        model_exists = os.path.exists("/app/model/flask_random_forest.rds")
        if not model_exists:
            return jsonify({
                "status": "unhealthy", 
                "message": "Model file not found",
                "model_loaded": False
            }), 503
            
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "version": "1.0.0"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        logger.info("📊 Received prediction request")
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Handle different input formats
        if 'instances' in data:
            # MLflow format
            instance = data['instances'][0]
        elif 'data' in data:
            # Seldon format  
            if 'ndarray' in data['data']:
                values = data['data']['ndarray'][0]
                instance = dict(zip(feature_names, values))
            else:
                instance = data['data']
        else:
            # Direct format
            instance = data
            
        logger.info(f"Processing input: {instance}")
        
        # Validate required features
        for feature in feature_names:
            if feature not in instance:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
        
        # Run R prediction
        try:
            # Run R script with input
            process = subprocess.run(
                ['Rscript', '-e', r_prediction_script],
                input=json.dumps(instance),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if process.returncode != 0:
                logger.error(f"R script error: {process.stderr}")
                return jsonify({"error": f"R prediction failed: {process.stderr}"}), 500
            
            # Parse R output
            try:
                result = json.loads(process.stdout.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, Output: {process.stdout}")
                return jsonify({"error": "Invalid JSON from R script"}), 500
            
            # Format response
            response = {
                "predictions": [{
                    "probability": result["probability"],
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "input": instance
                }],
                "model_name": "loan-approval-rf",
                "model_version": "1.0.0"
            }
            
            logger.info(f"✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
            return jsonify(response), 200
            
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Prediction timeout"}), 504
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metadata', methods=['GET'])
def metadata():
    """Model metadata endpoint"""
    return jsonify({
        "name": "loan-approval-model",
        "versions": ["1.0.0"],
        "platform": "R + RandomForest",
        "inputs": [
            {"name": "age", "datatype": "INT", "shape": [1]},
            {"name": "income", "datatype": "INT", "shape": [1]},
            {"name": "education", "datatype": "INT", "shape": [1]},
            {"name": "experience", "datatype": "INT", "shape": [1]},
            {"name": "credit_score", "datatype": "INT", "shape": [1]}
        ],
        "outputs": [
            {"name": "prediction", "datatype": "STR", "shape": [1]},
            {"name": "probability", "datatype": "FP64", "shape": [1]},
            {"name": "confidence", "datatype": "FP64", "shape": [1]}
        ]
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        "message": "Loan Approval Model Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "predict": "POST /predict - Make predictions",
            "metadata": "GET /metadata - Model information"
        },
        "example_request": {
            "age": 35,
            "income": 75000,
            "education": 16,
            "experience": 10,
            "credit_score": 750
        }
    }), 200

if __name__ == '__main__':
    logger.info("🚀 Starting Loan Approval Model Server")
    logger.info("📋 Endpoints available:")
    logger.info("   GET  /health   - Health check")
    logger.info("   POST /predict  - Make predictions") 
    logger.info("   GET  /metadata - Model information")
    logger.info("   GET  /         - API documentation")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=9000, debug=False)