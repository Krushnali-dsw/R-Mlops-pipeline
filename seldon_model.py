#!/usr/bin/env python3
"""
Seldon Core Model Wrapper for R Random Forest
Properly implements Seldon Core interface
"""

import os
import json
import subprocess
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanApprovalModel:
    """
    Seldon Core compatible model class
    """
    
    def __init__(self):
        self.model_loaded = False
        self.feature_names = ["age", "income", "education", "experience", "credit_score"]
        logger.info("🚀 Initializing Loan Approval Model")
        
        # R prediction script
        self.r_script = '''
library(randomForest)
library(jsonlite)

# Load model
model <- readRDS("/app/model/flask_random_forest.rds")

# Read input from command line args
args <- commandArgs(trailingOnly = TRUE)
input_json <- args[1]
input_data <- fromJSON(input_json)

# Create data frame
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

cat(toJSON(result, auto_unbox = TRUE))
'''

    def predict(self, X, features_names=None):
        """
        Seldon Core predict method
        
        Args:
            X: Input data as numpy array or list
            features_names: Feature names (optional)
            
        Returns:
            numpy array with predictions
        """
        try:
            logger.info("📊 Processing prediction request")
            
            # Handle input format
            if isinstance(X, (list, np.ndarray)):
                if len(X.shape) == 1:
                    # Single prediction
                    values = X.tolist() if hasattr(X, 'tolist') else X
                else:
                    # Batch prediction - take first row for now
                    values = X[0].tolist() if hasattr(X[0], 'tolist') else X[0]
                    
                # Create input dictionary
                input_data = dict(zip(self.feature_names, values))
            else:
                raise ValueError("Unsupported input format")
            
            logger.info(f"Input data: {input_data}")
            
            # Execute R script
            try:
                process = subprocess.run(
                    ['Rscript', '-e', self.r_script, json.dumps(input_data)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if process.returncode != 0:
                    logger.error(f"R script error: {process.stderr}")
                    return np.array([[0.5, 0.5]])  # Default prediction
                
                # Parse result
                result = json.loads(process.stdout.strip())
                
                # Return probability array for Seldon Core
                prob_denied = 1.0 - result["probability"]
                prob_approved = result["probability"]
                
                logger.info(f"✅ Prediction: {result['prediction']} (prob: {prob_approved:.3f})")
                
                # Return as numpy array with shape (n_samples, n_classes)
                return np.array([[prob_denied, prob_approved]])
                
            except subprocess.TimeoutExpired:
                logger.error("R prediction timeout")
                return np.array([[0.5, 0.5]])
            except Exception as e:
                logger.error(f"R execution error: {e}")
                return np.array([[0.5, 0.5]])
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.array([[0.5, 0.5]])
    
    def health_status(self):
        """Health check for Seldon Core"""
        try:
            model_exists = os.path.exists("/app/model/flask_random_forest.rds")
            return {
                "status": "healthy" if model_exists else "unhealthy",
                "model_loaded": model_exists
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def init_metadata(self):
        """Metadata for Seldon Core"""
        return {
            "name": "loan-approval-model",
            "versions": ["1.0.0"],
            "platform": "seldon-core",
            "inputs": [
                {"name": "age", "datatype": "INT32", "shape": [1]},
                {"name": "income", "datatype": "INT32", "shape": [1]}, 
                {"name": "education", "datatype": "INT32", "shape": [1]},
                {"name": "experience", "datatype": "INT32", "shape": [1]},
                {"name": "credit_score", "datatype": "INT32", "shape": [1]}
            ],
            "outputs": [
                {"name": "probability", "datatype": "FP64", "shape": [2]}
            ]
        }

# Create the model instance for Seldon Core
# Seldon expects the class to have the same name as the module
class Model(LoanApprovalModel):
    pass

# This is what Seldon Core will import
Model = LoanApprovalModel