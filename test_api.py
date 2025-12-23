"""
Test script for the Diabetes Prediction API
Run this after starting the API server to verify it works correctly
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the root endpoint"""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_single_prediction():
    """Test single patient prediction"""
    print("Testing single prediction endpoint...")
    
    # Sample patient data
    patient_data = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction endpoint...")
    
    # Sample batch data
    batch_data = {
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
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Diabetes Prediction API - Test Suite")
    print("=" * 60)
    print()
    
    try:
        results = {
            "Health Check": test_health_check(),
            "Single Prediction": test_single_prediction(),
            "Batch Prediction": test_batch_prediction(),
            "Health Endpoint": test_health_endpoint()
        }
        
        print("=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print()
        if all_passed:
            print("All tests passed! ✓")
        else:
            print("Some tests failed. Please check the output above.")
        
        return all_passed
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print(f"Please make sure the server is running at {BASE_URL}")
        print("Start the server with: uvicorn main:app --reload")
        return False


if __name__ == "__main__":
    run_all_tests()
