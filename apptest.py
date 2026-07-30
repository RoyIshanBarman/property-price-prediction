from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
import atexit
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.models = {}
        self.data = None
        self.features = None
        self.X_encoded = None

    def load_models_and_data(self):
        try:
            
            self.models = {
                'xgb': joblib.load('xgb_model.pkl'),
                'rf': joblib.load('rf_model.pkl'),
                'scaler': joblib.load('scaler.pkl'),
                'stacked': joblib.load('stacked_model.pkl')
            }
            
            
            self.data = pd.read_csv("dataset.csv")
            
            
            self._engineer_features()
            self._generate_synthetic_features()
            
            
            self._prepare_feature_matrix()
            
            logger.info("Models and data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models/data: {str(e)}")
            raise

    def _engineer_features(self):
        """Create derived features from raw data"""
        self.data['Price_to_Income_Ratio'] = self.data['Price'] / self.data['Median_Income']
        self.data['Growth_vs_Infra'] = self.data['Growth_Projection'] * self.data['Infrastructure_Score']
        self.data['Total_Rooms'] = self.data['Bedrooms'] + self.data['Bathrooms']
        self.data['Appreciation_Rate'] = np.where(
            self.data['Historical_Appreciation'] == 0, 0, 
            (self.data['Price'] - self.data['Historical_Appreciation']) / self.data['Historical_Appreciation']
        )

    def _generate_synthetic_features(self):
        """Generate additional synthetic features"""
        np.random.seed(52)
        synthetic_features = {
            'Economic_Growth_Index': (0, 1),
            'Crime_Trend': (0, 1),
            'Public_Transport_Accessibility': (0, 1),
            'Flood_Risk_Trend': (0, 1),
            'Air_Quality_Trend': (0, 1),
            'Historical_Price_Trends': (0, 1)
        }
        
        for feature, bounds in synthetic_features.items():
            self.data[feature] = np.random.uniform(*bounds, len(self.data))

        self.data['Interest_Rate'] = np.random.uniform(0.01, 0.05, len(self.data))
        self.data['Population_Density'] = np.random.randint(1000, 10000, len(self.data))
        self.data['Walk_Score'] = np.random.randint(30, 100, len(self.data))
        self.data['Seasonality'] = np.random.choice([0, 1], len(self.data))
        self.data['Future_Infrastructure_Projects'] = np.random.choice([0, 1], len(self.data))
        self.data['Government_Policies'] = np.random.choice([0, 1], len(self.data))

    def _prepare_feature_matrix(self):
        self.features = [
            'Size_sqft', 'Bedrooms', 'Bathrooms', 'Crime_Rate', 'School_Rating', 
            'Rental_Yield', 'Price_per_sqft', 'Median_Income', 'Unemployment_Rate', 
            'Infrastructure_Score', 'Flood_Risk', 'Historical_Appreciation', 
            'Growth_Projection', 'Distance_to_Metro', 'Property_Age', 
            'Air_Quality_Index', 'Planned_Infrastructure', 'Total_Rooms',
            'Price_to_Income_Ratio', 'Growth_vs_Infra', 'Economic_Growth_Index', 
            'Interest_Rate', 'Population_Density', 'Walk_Score', 'Crime_Trend',
            'Public_Transport_Accessibility', 'Flood_Risk_Trend', 'Air_Quality_Trend',
            'Historical_Price_Trends', 'Seasonality', 'Future_Infrastructure_Projects',
            'Government_Policies', 'Location'
        ]
        
        X = self.data[self.features]
        self.X_encoded = pd.get_dummies(X, columns=['Location'], drop_first=True)


app = Flask(__name__)
CORS(app)


app_state = AppState()


def initialize_app():
    app_state.load_models_and_data()

initialize_app()


def cleanup():
    pass  

atexit.register(cleanup)


def handle_errors(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return wrapped


class LocationRequest(BaseModel):
    Location: str

class PropertyRecommendation(BaseModel):
    property_name: str
    actual_price: float
    predicted_price: float
    appreciation_rate: float
    recommendation: str
    bedrooms: int
    bathrooms: int
    size_sqft: float

@app.route("/predict", methods=["POST"])
@handle_errors
def predict_properties():
    """Get property investment recommendations for a location"""
    request_data = request.get_json()
    location = request_data.get("Location")
    
    if not location:
        return jsonify({"error": "Location is required"}), 400
    
    if location not in app_state.data['Location'].unique():
        return jsonify({"error": f"No properties found for location: {location}"}), 404

    location_data = app_state.data[app_state.data['Location'] == location]
    X_location = pd.get_dummies(
        location_data[app_state.features], 
        columns=['Location'], 
        drop_first=True
    )
    
    
    X_location = X_location.reindex(columns=app_state.X_encoded.columns, fill_value=0)
    X_location_scaled = app_state.models['scaler'].transform(X_location)
    
    
    stacked_location_preds = app_state.models['stacked'].predict(X_location_scaled)
    
    
    appreciation_rate = (
        (stacked_location_preds - location_data['Price'].values) / 
        location_data['Price'].values
    ) * 100  

    
    recommendations = np.select(
        [appreciation_rate > 30, appreciation_rate > 10, appreciation_rate > 0],
        ["Strong Buy", "Buy", "Hold"], 
        default="Don't Buy"
    )

    
    response = [
        {
            "property_name": row['Property_Name'],
            "actual_price": float(row['Price']),
            "predicted_price": float(pred),
            "appreciation_rate": float(rate),
            "recommendation": rec,
            "bedrooms": int(row['Bedrooms']),
            "bathrooms": int(row['Bathrooms']),
            "size_sqft": float(row['Size_sqft'])
        }
        for row, pred, rate, rec in zip(
            location_data.to_dict('records'),
            stacked_location_preds,
            appreciation_rate,
            recommendations
        )
    ]
    
    return jsonify(response)

@app.route("/locations", methods=["GET"])
@handle_errors
def get_available_locations():
    Location = [
    "Alipore",
    "Ballygunge",
    "Baranagar",
    "Baguiati",
    "Behala",
    "Chinar Park",
    "Dum Dum",
    "Garia",
    "Howrah",
    "Jadavpur",
    "Joka",
    "Kasba",
    "Lake Town",
    "New Town",
    "Park Street",
    "Rajarhat",
    "Salt Lake",
    "Sealdah",
    "Tollygunge",
    "Ultadanga"
]
    if app_state.data is None:
        return jsonify({"error": "Data not loaded yet"}), 503
    return jsonify(sorted(app_state.data['Location'].unique().tolist()))

@app.route("/", methods=["GET"])
@handle_errors
def root():
    return jsonify({"message": "Property Investment Advisor API is running"})

if __name__ == "__main__":
    app.run(debug=True)