import random
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin


random.seed(52)
np.random.seed(52)


data = pd.read_csv("dataset.csv")


data['Price_to_Income_Ratio'] = data['Price'] / data['Median_Income']
data['Growth_vs_Infra'] = data['Growth_Projection'] * data['Infrastructure_Score']
data['Total_Rooms'] = data['Bedrooms'] + data['Bathrooms']
data['Appreciation_Rate'] = np.where(
    data['Historical_Appreciation'] == 0, 0, 
    (data['Price'] - data['Historical_Appreciation']) / data['Historical_Appreciation']
)


np.random.seed(52)
for feature, bounds in {
    'Economic_Growth_Index': (0, 1),
    'Crime_Trend': (0, 1),
    'Public_Transport_Accessibility': (0, 1),
    'Flood_Risk_Trend': (0, 1),
    'Air_Quality_Trend': (0, 1),
    'Historical_Price_Trends': (0, 1)
}.items():
    data[feature] = np.random.uniform(*bounds, len(data))

data['Interest_Rate'] = np.random.uniform(0.01, 0.05, len(data))
data['Population_Density'] = np.random.randint(1000, 10000, len(data))
data['Walk_Score'] = np.random.randint(30, 100, len(data))
data['Seasonality'] = np.random.choice([0, 1], len(data))
data['Future_Infrastructure_Projects'] = np.random.choice([0, 1], len(data))
data['Government_Policies'] = np.random.choice([0, 1], len(data))


features = [
    'Size_sqft', 'Bedrooms', 'Bathrooms', 'Crime_Rate', 'School_Rating', 'Rental_Yield',
    'Price_per_sqft', 'Median_Income', 'Unemployment_Rate', 'Infrastructure_Score',
    'Flood_Risk', 'Historical_Appreciation', 'Growth_Projection', 'Distance_to_Metro',
    'Property_Age', 'Air_Quality_Index', 'Planned_Infrastructure', 'Total_Rooms',
    'Price_to_Income_Ratio', 'Growth_vs_Infra', 'Economic_Growth_Index', 'Interest_Rate',
    'Population_Density', 'Walk_Score', 'Crime_Trend', 'Public_Transport_Accessibility',
    'Flood_Risk_Trend', 'Air_Quality_Trend', 'Historical_Price_Trends', 'Seasonality',
    'Future_Infrastructure_Projects', 'Government_Policies', 'Location'
]
X = data[features]
y = data['Price']


X_encoded = pd.get_dummies(X, columns=['Location'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


xgb_model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    subsample=0.8, 
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


stacked_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    final_estimator=xgb.XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3),
    cv=5
)


stacked_model.fit(X_train_scaled, y_train)


joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(stacked_model, 'stacked_model.pkl')

def get_property_recommendations(location):
    if location not in data['Location'].unique():
        print(f"No properties found for location: {location}")
        return None

    location_data = data[data['Location'] == location]
    X_location = pd.get_dummies(location_data[features], columns=['Location'], drop_first=True)
    X_location = X_location.reindex(columns=X_encoded.columns, fill_value=0)
    X_location_scaled = scaler.transform(X_location)
    stacked_location_preds = stacked_model.predict(X_location_scaled)
    appreciation_rate = (stacked_location_preds - location_data['Price'].values) / location_data['Price'].values

    recommendations = np.select(
        [appreciation_rate > 0.3, appreciation_rate > 0.1, appreciation_rate > 0],
        ["Strong Buy", "Buy", "Hold"], 
        default="Don't Buy"
    )

    results_df = pd.DataFrame({
        'Property_Name': location_data['Property_Name'].values,
        'Actual_Price': location_data['Price'].values,
        'Predicted_Price': stacked_location_preds,
        'Appreciation_Rate': appreciation_rate,
        'Recommendation': recommendations
    })

    return results_df


location = input("Enter the location: ").strip()
results = get_property_recommendations(location)

if results is not None:
    print(results)
    results.to_csv(f'{location}_property_investment_recommendations.csv', index=False)