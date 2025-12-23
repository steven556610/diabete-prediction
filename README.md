# Diabetes Prediction API

A FastAPI-based REST API for predicting diabetes probability using a trained XGBoost model.

## Features

- **Single Prediction**: Predict diabetes probability for one patient
- **Batch Prediction**: Predict for multiple patients at once
- **Docker Support**: Easy deployment with Docker
- **Auto-generated API Docs**: Interactive documentation at `/docs`

## Project Structure

```
diabete_prediction_challenge/
├── code/
│   └── core_logic.py       # Core prediction logic
├── model/
│   └── xgb_numeric_only.pkl # Trained XGBoost model
├── data/                    # Training/test data (not included in Docker)
├── notebook/                # Jupyter notebooks (not included in Docker)
├── main.py                  # FastAPI application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
└── .dockerignore           # Docker ignore file
```

## Installation

### Option 1: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

1. Build the Docker image:
```bash
docker build -t diabetes-prediction-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 diabetes-prediction-api
```

### Option 3: Docker Compose (Recommended)

1. Start the service:
```bash
docker-compose up -d
```

2. Stop the service:
```bash
docker-compose down
```

## API Usage

Once running, the API will be available at `http://localhost:8000`

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

### Endpoints

#### 1. Health Check
```bash
GET /
```

Response:
```json
{
  "message": "Diabetes Prediction API is running!",
  "status": "healthy",
  "model": "XGBoost (numeric features only)"
}
```

#### 2. Single Prediction
```bash
POST /predict
```

Request body:
```json
{
  "age": 45,
  "alcohol_consumption_per_week": 2,
  "physical_activity_minutes_per_week": 150,
  "diet_score": 7.5,
  "sleep_hours_per_day": 7,
  "screen_time_hours_per_day": 4,
  "bmi": 28.5,
  "waist_to_hip_ratio": 0.85,
  "systolic_bp": 130,
  "diastolic_bp": 85,
  "heart_rate": 75,
  "cholesterol_total": 200,
  "hdl_cholesterol": 50,
  "ldl_cholesterol": 120,
  "triglycerides": 150,
  "family_history_diabetes": 1,
  "hypertension_history": 0,
  "cardiovascular_history": 0
}
```

Response:
```json
{
  "probability": 0.65,
  "id": 0
}
```

#### 3. Batch Prediction
```bash
POST /predict/batch
```

Request body:
```json
{
  "patients": [
    {
      "age": 45,
      "alcohol_consumption_per_week": 2,
      "physical_activity_minutes_per_week": 150,
      "diet_score": 7.5,
      "sleep_hours_per_day": 7,
      "screen_time_hours_per_day": 4,
      "bmi": 28.5,
      "waist_to_hip_ratio": 0.85,
      "systolic_bp": 130,
      "diastolic_bp": 85,
      "heart_rate": 75,
      "cholesterol_total": 200,
      "hdl_cholesterol": 50,
      "ldl_cholesterol": 120,
      "triglycerides": 150,
      "family_history_diabetes": 1,
      "hypertension_history": 0,
      "cardiovascular_history": 0
    },
    {
      "age": 52,
      "alcohol_consumption_per_week": 3,
      "physical_activity_minutes_per_week": 90,
      "diet_score": 5.5,
      "sleep_hours_per_day": 6,
      "screen_time_hours_per_day": 6,
      "bmi": 31.2,
      "waist_to_hip_ratio": 0.92,
      "systolic_bp": 145,
      "diastolic_bp": 92,
      "heart_rate": 82,
      "cholesterol_total": 240,
      "hdl_cholesterol": 42,
      "ldl_cholesterol": 160,
      "triglycerides": 190,
      "family_history_diabetes": 1,
      "hypertension_history": 1,
      "cardiovascular_history": 0
    }
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "probability": 0.65,
      "id": 0
    },
    {
      "probability": 0.78,
      "id": 1
    }
  ]
}
```

## Model Information

The model uses **only numeric features** for prediction:

### Required Features:
- `age`: Age of the patient
- `alcohol_consumption_per_week`: Alcohol consumption per week
- `physical_activity_minutes_per_week`: Physical activity minutes per week
- `diet_score`: Diet quality score
- `sleep_hours_per_day`: Sleep hours per day
- `screen_time_hours_per_day`: Screen time hours per day
- `bmi`: Body Mass Index
- `waist_to_hip_ratio`: Waist to hip ratio
- `systolic_bp`: Systolic blood pressure
- `diastolic_bp`: Diastolic blood pressure
- `heart_rate`: Heart rate
- `cholesterol_total`: Total cholesterol
- `hdl_cholesterol`: HDL cholesterol
- `ldl_cholesterol`: LDL cholesterol
- `triglycerides`: Triglycerides level
- `family_history_diabetes`: Family history of diabetes (0 or 1)
- `hypertension_history`: Hypertension history (0 or 1)
- `cardiovascular_history`: Cardiovascular history (0 or 1)

### Optional Features (ignored by model):
- `gender`, `ethnicity`, `education_level`, `income_level`, `smoking_status`, `employment_status`

## Example with cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "alcohol_consumption_per_week": 2,
    "physical_activity_minutes_per_week": 150,
    "diet_score": 7.5,
    "sleep_hours_per_day": 7,
    "screen_time_hours_per_day": 4,
    "bmi": 28.5,
    "waist_to_hip_ratio": 0.85,
    "systolic_bp": 130,
    "diastolic_bp": 85,
    "heart_rate": 75,
    "cholesterol_total": 200,
    "hdl_cholesterol": 50,
    "ldl_cholesterol": 120,
    "triglycerides": 150,
    "family_history_diabetes": 1,
    "hypertension_history": 0,
    "cardiovascular_history": 0
  }'
```

## Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

MIT
