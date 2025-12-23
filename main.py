from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from code.core_logic import load_model, predict_single, predict_batch

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes probability using XGBoost model",
    version="1.0.0"
)

# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    global model
    try:
        model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Define input data models
class PatientData(BaseModel):
    """Single patient data for diabetes prediction"""
    age: float = Field(..., description="Age of the patient")
    alcohol_consumption_per_week: float = Field(..., description="Alcohol consumption per week")
    physical_activity_minutes_per_week: float = Field(..., description="Physical activity minutes per week")
    diet_score: float = Field(..., description="Diet quality score")
    sleep_hours_per_day: float = Field(..., description="Sleep hours per day")
    screen_time_hours_per_day: float = Field(..., description="Screen time hours per day")
    bmi: float = Field(..., description="Body Mass Index")
    waist_to_hip_ratio: float = Field(..., description="Waist to hip ratio")
    systolic_bp: float = Field(..., description="Systolic blood pressure")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure")
    heart_rate: float = Field(..., description="Heart rate")
    cholesterol_total: float = Field(..., description="Total cholesterol")
    hdl_cholesterol: float = Field(..., description="HDL cholesterol")
    ldl_cholesterol: float = Field(..., description="LDL cholesterol")
    triglycerides: float = Field(..., description="Triglycerides level")
    family_history_diabetes: int = Field(..., description="Family history of diabetes (0 or 1)")
    hypertension_history: int = Field(..., description="Hypertension history (0 or 1)")
    cardiovascular_history: int = Field(..., description="Cardiovascular history (0 or 1)")
    
    # Optional categorical features (will be dropped by the model)
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    education_level: Optional[str] = None
    income_level: Optional[str] = None
    smoking_status: Optional[str] = None
    employment_status: Optional[str] = None


class BatchPatientData(BaseModel):
    """Batch of patient data for diabetes prediction"""
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    probability: float = Field(..., description="Probability of diabetes (0-1)")
    id: Optional[int] = Field(None, description="Patient ID if provided")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse]


# API Endpoints
@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "Diabetes Prediction API is running!",
        "status": "healthy",
        "model": "XGBoost (numeric features only)"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    """
    Predict diabetes probability for a single patient.
    
    Returns the probability of diabetes (0-1) based on patient features.
    """
    try:
        # Convert Pydantic model to dictionary
        data_dict = data.dict(exclude_none=True)
        
        # Make prediction
        result = predict_single(data_dict, model=model)
        
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(data: BatchPatientData):
    """
    Predict diabetes probability for multiple patients.
    
    Returns a list of predictions with probabilities for each patient.
    """
    try:
        # Convert list of Pydantic models to list of dictionaries
        data_list = [patient.dict(exclude_none=True) for patient in data.patients]
        
        # Make batch prediction
        results = predict_batch(data_list, model=model)
        
        # Convert to response model
        predictions = [PredictionResponse(**result) for result in results]
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }