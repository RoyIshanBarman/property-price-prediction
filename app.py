from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained models and scaler
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('xgb_model.pkl')
nn_model = load_model('nn_model.h5')
stacked_model = joblib.load('stacked_model_final_estimator.pkl')  # Assumes you saved stacked_model as well

# Load the dataset for location-based filtering
data = pd.read_csv('dataset.csv')

# Initialize FastAPI
app = FastAPI()

# Define the request model
class LocationRequest(BaseModel):
    location: str

# Preprocessing function
def preprocess_data(location):
    # Filter data by location
    data_location = data[data['Location'] == location]
    
    if data_location.empty:
        return None, None  # Return early if no data for location

    # Reuse the feature engineering logic
    data_location['Price_to_Income_Ratio'] = data_location['Price'] / data_location['Median_Income']
    data_location['Growth_vs_Infra'] = data_location['Growth_Projection'] * data_location['Infrastructure_Score']
    data_location['Total_Rooms'] = data_location['Bedrooms'] + data_location['Bathrooms']
    data_location['Appreciation_Rate'] = (data_location['Price'] - data_location['Historical_Appreciation']) / data_location['Historical_Appreciation']
    
    # Add other features
    additional_features = [
        'Economic_Growth_Index', 'Interest_Rate', 'Population_Density', 'Walk_Score', 
        'Crime_Trend', 'Public_Transport_Accessibility', 'Flood_Risk_Trend', 
        'Air_Quality_Trend', 'Historical_Price_Trends', 'Seasonality', 
        'Future_Infrastructure_Projects', 'Government_Policies'
    ]
    data_location[additional_features] = np.random.random(size=(len(data_location), len(additional_features)))

    # Select features and one-hot encode
    features = data_location.drop(columns=['Property_Name', 'Price'])
    features = pd.get_dummies(features, columns=['Location'], drop_first=True)
    
    # Reindex to ensure all required columns are present
    trained_feature_names = stacked_model.named_estimators_['xgb'].get_booster().feature_names
    X = features.reindex(columns=trained_feature_names, fill_value=0)

    # Scale features
    X_scaled = scaler.transform(X)

    return X_scaled, data_location

# Recommendation function
def recommend_appreciation(rate):
    if rate > 0.3:
        return "Strong Buy"
    elif rate > 0.1:
        return "Buy"
    elif rate > 0:
        return "Hold"
    else:
        return "Don't Buy"

# Define the endpoint
@app.post("/predict")
async def predict_properties(location_request: LocationRequest):
    location = location_request.location
    X_scaled, data_location = preprocess_data(location)
    
    if X_scaled is None or data_location.empty:
        raise HTTPException(status_code=404, detail="No properties found for the specified location.")

    # Make predictions
    predictions = stacked_model.predict(X_scaled)
    appreciation_rates = (predictions - data_location['Price'].values) / data_location['Price'].values
    recommendations = [recommend_appreciation(rate) for rate in appreciation_rates]

    # Prepare response
    results = []
    for i, property_name in enumerate(data_location['Property_Name'].values):
        results.append({
            "Property_Name": property_name,
            "Actual_Price": data_location['Price'].values[i],
            "Predicted_Price": predictions[i],
            "Appreciation_Rate": appreciation_rates[i],
            "Recommendation": recommendations[i]
        })

    return {"properties": results}
