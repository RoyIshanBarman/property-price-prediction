from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
import json

app = FastAPI()

# Load the pre-trained models and scaler
stacked_model = joblib.load('stacked_model.pkl')
scaler = joblib.load('scaler.pkl')

class Location(BaseModel):
    location: str

def get_property_recommendations(location):
    data = pd.read_csv('dataset.csv')
    location_data = data[data['Location'] == location]

    if location_data.empty:
        return "No properties found for this location."

    features = [
        'Size_sqft', 'Bedrooms', 'Bathrooms', 'Crime_Rate', 'School_Rating',
        'Rental_Yield', 'Price_per_sqft', 'Median_Income', 'Unemployment_Rate',
        'Infrastructure_Score', 'Flood_Risk', 'Historical_Appreciation',
        'Growth_Projection', 'Distance_to_Metro', 'Property_Age', 'Air_Quality_Index',
        'Planned_Infrastructure', 'Total_Rooms', 'Price_to_Income_Ratio',
        'Growth_vs_Infra', 'Economic_Growth_Index', 'Interest_Rate',
        'Population_Density', 'Walk_Score', 'Crime_Trend',
        'Public_Transport_Accessibility', 'Flood_Risk_Trend', 'Air_Quality_Trend',
        'Historical_Price_Trends', 'Seasonality', 'Future_Infrastructure_Projects',
        'Government_Policies', 'Location'
    ]

    X_location = pd.get_dummies(location_data[features], columns=['Location'], drop_first=True)
    X_encoded_cols = pd.DataFrame(columns=X_encoded.columns)  # Dummy for column matching
    X_location = X_location.reindex(columns=X_encoded_cols.columns, fill_value=0)

    X_location_scaled = scaler.transform(X_location)

    predictions = stacked_model.predict(X_location_scaled)
    appreciation_rate = (predictions - location_data['Price'].values) / location_data['Price'].values

    def recommend_appreciation(rate):
        if rate > 0.3:
            return "Strong Buy"
        elif rate > 0.1:
            return "Buy"
        elif rate > 0:
            return "Hold"
        else:
            return "Don't Buy"

    recommendations = [recommend_appreciation(rate) for rate in appreciation_rate]
    results_df = pd.DataFrame({
        'Property_Name': location_data['Property_Name'].values,
        'Actual_Price': location_data['Price'].values,
        'Predicted_Price': predictions,
        'Appreciation_Rate': appreciation_rate,
        'Recommendation': recommendations
    })
    return results_df

@app.post("/predict")
def predict(location: Location):
    results = get_property_recommendations(location.location)
    if isinstance(results, str):
        return {"error": results}, 404
    return json.loads(results.to_json(orient="records"))