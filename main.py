import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import StackingRegressor
import joblib

random.seed(52)


data = pd.read_csv('dataset.csv')


data['Price_to_Income_Ratio'] = data['Price'] / data['Median_Income']
data['Growth_vs_Infra'] = data['Growth_Projection'] * data['Infrastructure_Score']
data['Total_Rooms'] = data['Bedrooms'] + data['Bathrooms']
data['Appreciation_Rate'] = (data['Price'] - data['Historical_Appreciation']) / data['Historical_Appreciation']


data['Economic_Growth_Index'] = np.random.random(size=len(data))
data['Interest_Rate'] = np.random.uniform(0.01, 0.05, len(data))
data['Population_Density'] = np.random.randint(1000, 10000, len(data))
data['Walk_Score'] = np.random.randint(30, 100, len(data))
data['Crime_Trend'] = np.random.random(size=len(data))
data['Public_Transport_Accessibility'] = np.random.random(size=len(data))
data['Flood_Risk_Trend'] = np.random.random(size=len(data))
data['Air_Quality_Trend'] = np.random.random(size=len(data))
data['Historical_Price_Trends'] = np.random.random(size=len(data))
data['Seasonality'] = np.random.choice([0, 1], len(data))
data['Future_Infrastructure_Projects'] = np.random.choice([0, 1], len(data))
data['Government_Policies'] = np.random.choice([0, 1], len(data))


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
X = data[features]
y = data['Price']


X_encoded = pd.get_dummies(X, columns=['Location'], drop_first=True)


property_names = data['Property_Name']


X_train, X_test, y_train, y_test, property_names_train, property_names_test = train_test_split(
    X_encoded, y, property_names, test_size=0.2, random_state=42)


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


def create_nn_model():
    nn_model = Sequential()
    nn_model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
    for _ in range(13):  
        nn_model.add(Dense(64, activation='relu'))
    nn_model.add(Dense(1))  
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    return nn_model

# Train and save the neural network model separately
nn_model_instance = create_nn_model()
nn_model_instance.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
save_model(nn_model_instance, 'nn_model.h5')

# Wrap the neural network model with KerasRegressor for stacking
nn_model = KerasRegressor(model=create_nn_model, epochs=100, batch_size=32, verbose=0)

# Stack models using StackingRegressor
stacked_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('nn', nn_model)],
    final_estimator=xgb.XGBRegressor()
)

# Train the stacked model
stacked_model.fit(X_train_scaled, y_train)

# Save the XGBoost model and scaler
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions
stacked_test_preds = stacked_model.predict(X_test_scaled)

# Calculate appreciation rate
appreciation_rate = (stacked_test_preds - y_test.values) / y_test.values

# Generate recommendations based on appreciation rates
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

# Prepare the results DataFrame
results_df = pd.DataFrame({
    'Property_Name': property_names_test.values,
    'Actual_Price': y_test.values,
    'Predicted_Price': stacked_test_preds,
    'Appreciation_Rate': appreciation_rate,
    'Recommendation': recommendations
})


print(results_df.head())

results_df.to_csv('property_investment_recommendations.csv', index=False)
