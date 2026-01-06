"""
Customer Segmentation API

This module provides a FastAPI endpoint for customer segmentation predictions.
Includes real-time prediction, batch processing, and model management.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import json

# Import our modules
from config import get_file_path, KMEANS_MODEL_FILE, SCALER_MODEL_FILE
from data_validation import DataValidator
from model_evaluation import ModelEvaluator


# Pydantic models for API
class CustomerFeatures(BaseModel):
    """Customer features for single prediction."""
    CustomerID: str = Field(..., description="Unique customer identifier (numeric or NXXXX format)")
    Recency: int = Field(..., ge=0, description="Days since last purchase")
    Frequency: int = Field(..., ge=0, description="Number of purchases")
    Monetary: float = Field(..., ge=0, description="Total monetary value")
    TotalItems: Optional[int] = Field(None, ge=0, description="Total items purchased")
    UniqueProducts: Optional[int] = Field(None, ge=0, description="Number of unique products")
    Country: Optional[str] = Field(None, description="Customer country")
    AvgOrderValue: Optional[float] = Field(None, ge=0, description="Average order value")
    ItemsPerOrder: Optional[float] = Field(None, ge=0, description="Items per order")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    customers: List[CustomerFeatures] = Field(..., description="List of customers to segment")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    CustomerID: str
    Cluster: int
    Segment: str
    Confidence: Optional[float] = None
    Timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time: float
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    last_trained: Optional[str] = None
    n_clusters: int
    features_used: List[str]
    accuracy_metrics: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str


# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="API for customer segmentation using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
scaler = None
model_evaluator = None
validator = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = {}
        self.last_loaded = None
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            # Load K-means model
            model_path = get_file_path('kmeans_model')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load scaler
            scaler_path = get_file_path('scaler_model')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            # Update model info
            self.model_info = {
                'model_name': 'KMeans Customer Segmentation',
                'version': '1.0.0',
                'last_trained': datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).isoformat() if os.path.exists(model_path) else None,
                'n_clusters': self.model.n_clusters,
                'features_used': ['Recency', 'Frequency', 'Monetary']
            }
            
            self.last_loaded = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_single(self, features: CustomerFeatures) -> Dict:
        """Make prediction for single customer."""
        if not self.model or not self.scaler:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        try:
            # Prepare features
            feature_array = np.array([[
                features.Recency,
                features.Frequency,
                features.Monetary
            ]])
            
            # Scale features
            scaled_features = self.scaler.transform(feature_array)
            
            # Predict cluster
            cluster = self.model.predict(scaled_features)[0]
            
            # Get segment name (simplified mapping)
            segment_mapping = {
                0: 'Bargain Hunters',
                1: 'Loyal High-Spenders', 
                2: 'At-Risk Customers'
            }
            segment = segment_mapping.get(cluster, f'Cluster {cluster}')
            
            # Calculate confidence (distance to cluster center)
            distances = self.model.transform(scaled_features)[0]
            confidence = 1 / (1 + distances[cluster])  # Inverse distance
            
            return {
                'CustomerID': features.CustomerID,
                'Cluster': int(cluster),
                'Segment': segment,
                'Confidence': float(confidence),
                'Timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, customers: List[CustomerFeatures]) -> List[Dict]:
        """Make predictions for multiple customers."""
        predictions = []
        
        for customer in customers:
            try:
                prediction = self.predict_single(customer)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Batch prediction error for {customer.CustomerID}: {e}")
                # Add error prediction
                predictions.append({
                    'CustomerID': customer.CustomerID,
                    'Cluster': -1,
                    'Segment': 'Error',
                    'Confidence': 0.0,
                    'Timestamp': datetime.now().isoformat(),
                    'Error': str(e)
                })
        
        return predictions


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("Starting Customer Segmentation API")
    
    # Load models
    if model_manager.load_models():
        logger.info("Models loaded successfully")
    else:
        logger.error("Failed to load models")
    
    # Initialize validator and evaluator
    global validator, model_evaluator
    validator = DataValidator()
    model_evaluator = ModelEvaluator()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Customer Segmentation API")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Segmentation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.model else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.model is not None,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_manager.model_info)


@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single_customer(customer: CustomerFeatures):
    """Predict segment for single customer."""
    logger.info(f"Single prediction request for customer: {customer.CustomerID}")
    
    prediction = model_manager.predict_single(customer)
    return PredictionResponse(**prediction)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_customers(request: BatchPredictionRequest):
    """Predict segments for multiple customers."""
    start_time = datetime.now()
    logger.info(f"Batch prediction request for {len(request.customers)} customers")
    
    predictions = model_manager.predict_batch(request.customers)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**pred) for pred in predictions],
        total_processed=len(predictions),
        processing_time=processing_time,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict segments from uploaded CSV file."""
    logger.info(f"File prediction request: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert to CustomerFeatures objects
        customers = []
        for _, row in df.iterrows():
            customers.append(CustomerFeatures(**row.to_dict()))
        
        # Make predictions
        predictions = model_manager.predict_batch(customers)
        
        # Create response dataframe
        response_df = pd.DataFrame(predictions)
        
        # Return as JSON
        return {
            "predictions": predictions,
            "total_processed": len(predictions),
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.post("/validate/data")
async def validate_customer_data(customers: List[CustomerFeatures]):
    """Validate customer data."""
    logger.info(f"Data validation request for {len(customers)} customers")
    
    # Convert to DataFrame
    df = pd.DataFrame([customer.dict() for customer in customers])
    
    # Validate using our validator
    if validator:
        validation_results = validator.validate_customer_features(df)
        
        return {
            "validation_status": validation_results['status'],
            "issues": validation_results['issues'],
            "warnings": validation_results['warnings'],
            "statistics": validation_results['statistics'],
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {"error": "Validator not initialized"}


@app.get("/segments/summary")
async def get_segments_summary():
    """Get summary of customer segments."""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Define segment characteristics (simplified)
    segments = {
        0: {
            "name": "Bargain Hunters",
            "description": "Price-sensitive customers with low frequency",
            "characteristics": ["Low monetary value", "Infrequent purchases", "Price conscious"],
            "recommended_actions": ["Discount campaigns", "Price comparison highlights", "Value messaging"]
        },
        1: {
            "name": "Loyal High-Spenders",
            "description": "High-value customers with frequent purchases",
            "characteristics": ["High monetary value", "High frequency", "Brand loyal"],
            "recommended_actions": ["VIP programs", "Early access offers", "Personalized service"]
        },
        2: {
            "name": "At-Risk Customers",
            "description": "Customers who haven't purchased recently",
            "characteristics": ["High recency", "Low frequency", "Churn risk"],
            "recommended_actions": ["Re-engagement campaigns", "Special offers", "Win-back strategies"]
        }
    }
    
    return {
        "segments": segments,
        "total_segments": len(segments),
        "model_info": model_manager.model_info,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/models/reload")
async def reload_models():
    """Reload models from disk."""
    logger.info("Model reload request")
    
    if model_manager.load_models():
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload models")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
