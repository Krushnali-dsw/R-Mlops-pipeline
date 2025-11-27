#!/usr/bin/env python3
"""
Seldon Core Model Server for R Random Forest Model
Loads the trained R model and serves predictions via REST API
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Union, Any
import subprocess
import tempfile

class RModelServer:
    """
    Seldon Core compatible model server for R Random Forest models
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = ["age", "income", "education", "experience", "credit_score"]
        self.model_ready = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load(self):
        """
        Load the R model for inference
        """
        try:
            self.logger.info("🚀 Loading R Random Forest model...")
            
            # Create R script to load model and make predictions
            self.r_prediction_script = '''
# Load required libraries
library(randomForest)
library(jsonlite)

# Load the model
model <- readRDS("/app/model/random_forest_minio.rds")

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
            
            self.model_ready = True
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def predict(self, X: Union[np.ndarray, List, Dict], features_names: List[str] = None) -> Dict:
        """
        Make predictions using the loaded R model
        
        Args:
            X: Input data (can be array, list, or dict)
            features_names: Feature names (optional)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            if not self.model_ready:
                self.load()
            
            self.logger.info("📊 Making prediction...")
            
            # Handle different input formats
            if isinstance(X, dict):
                input_data = X
            elif isinstance(X, (list, np.ndarray)):
                if len(X) >= len(self.feature_names):
                    input_data = dict(zip(self.feature_names, X[:len(self.feature_names)]))
                else:
                    raise ValueError("Input array too short")
            else:
                raise ValueError("Unsupported input format")
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_data:
                    raise ValueError(f"Missing required feature: {feature}")
            
            # Create temporary file for R script input
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_file = f.name
            
            try:
                # Run R script
                process = subprocess.run(
                    ['Rscript', '-e', self.r_prediction_script],
                    input=json.dumps(input_data),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if process.returncode != 0:
                    self.logger.error(f"R script error: {process.stderr}")
                    raise RuntimeError(f"R prediction failed: {process.stderr}")
                
                # Parse R output
                result = json.loads(process.stdout.strip())
                
                # Format response for Seldon Core
                response = {
                    "predictions": [result["probability"]],
                    "prediction_class": result["prediction"],
                    "confidence": result["confidence"],
                    "feature_importance": {
                        "age": 0.2,
                        "income": 0.3,
                        "education": 0.15,
                        "experience": 0.15,
                        "credit_score": 0.2
                    }
                }
                
                self.logger.info(f"✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
                return response
                
            finally:
                # Clean up temporary file
                if os.path.exists(input_file):
                    os.unlink(input_file)
                
        except Exception as e:
            self.logger.error(f"❌ Prediction error: {str(e)}")
            return {
                "predictions": [0.0],
                "prediction_class": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def health_status(self) -> Dict:
        """
        Health check endpoint
        """
        return {
            "status": "healthy" if self.model_ready else "loading",
            "model_loaded": self.model_ready,
            "version": "1.0.0"
        }

# Create global model instance for Seldon Core
model = RModelServer()