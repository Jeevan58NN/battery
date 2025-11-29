from custom_pinn import PINN, ResidualBlock
from fastapi import FastAPI, Header, HTTPException
import tensorflow as tf
import joblib
import numpy as np
import json
import os
import logging
import uvicorn
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PINN Inference API")

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY environment variable not set! API will be unsecured.")

# Global variables for model and metadata
model = None
meta = None
templates = None

@app.on_event("startup")
async def startup_event():
    """Load model and data files on startup"""
    global model, meta, templates
    
    try:
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")

        # Load model
        logger.info("Loading PINN model...")

        custom_objects = {
            "PINN": PINN,
            "ResidualBlock": ResidualBlock
        }

        model = tf.keras.models.load_model(
            "pinn_inference.keras",
            compile=False,
            custom_objects=custom_objects
        )
        logger.info("✓ Model loaded successfully")
        
        # Load metadata
        logger.info("Loading metadata...")
        meta = joblib.load("pinn_meta_full.pkl")
        logger.info("✓ Metadata loaded successfully")
        
        # Load templates
        logger.info("Loading material templates...")
        with open("material_templates.json", "r") as f:
            templates = json.load(f)
        logger.info("✓ Templates loaded successfully")
        
        logger.info("All resources loaded. API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load resources: {str(e)}")
        raise


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "PINN Inference API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "metadata_loaded": meta is not None,
        "templates_loaded": templates is not None
    }


@app.post("/predict")
def predict(features: dict, x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Prediction endpoint
    
    Headers:
        X-API-Key: Your API key
        
    Body:
        JSON object with feature names and values
        Example: {"feature1": 0.5, "feature2": 1.2, ...}
    """
    # Check if API key is required and validate
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=403, 
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    
    # Check if model is loaded
    if model is None or meta is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service temporarily unavailable."
        )
    
    try:
        # Extract features in correct order
        x = []
        missing_features = []
        
        for name in meta["feature_names"]:
            if name not in features:
                missing_features.append(name)
            else:
                x.append(features[name])
        
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Prepare input and predict
        x = np.array(x).reshape(1, -1)
        y = model.predict(x, verbose=0)
        
        return {
            "output": float(y[0][0]),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/info")
def info():
    """Get information about required features"""
    if meta is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata not loaded"
        )
    
    return {
        "feature_names": meta.get("feature_names", []),
        "num_features": len(meta.get("feature_names", [])),
        "templates_available": templates is not None
    }


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
