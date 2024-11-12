import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('property_investment_recommendations.csv')

sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'axes.titlesize': 16, 'axes.labelsize': 12})


def plot_property_analysis(results_df, num_properties=5):
    """
    Plot analysis for a subset of properties.
    Parameters:
        results_df: DataFrame containing property analysis results
        num_properties: Number of properties to visualize
    """
    subset_df = results_df.head(num_properties)  

    for index, row in subset_df.iterrows():
        plt.figure(figsize=(12, 6))
        plt.plot(['Actual Price', 'Predicted Price'], 
                 [row['Actual_Price'], row['Predicted_Price']], 
                 marker='o', color='b', label='Price')
        
        plt.title(f"Actual vs Predicted Price for {row['Property_Name']}")
        plt.ylabel("Price")
        plt.xlabel("Price Type")
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(['Initial', 'Predicted'], 
                 [0, row['Appreciation_Rate']], 
                 marker='o', linestyle='--', color='r', label='Appreciation Rate')
        
        plt.title(f"Appreciation Rate Trend for {row['Property_Name']}")
        plt.ylabel("Appreciation Rate")
        plt.xlabel("Timeframe")
        plt.legend()
        plt.show()

plot_property_analysis(results_df, num_properties=5)
