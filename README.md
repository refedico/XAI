# XAI with SHAP and LIME on California Housing Prices

## Overview
This project demonstrates the use of Explainable AI (XAI) techniques, specifically SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), to explain the predictions of a machine learning model trained on the California Housing Prices dataset. The goal is to provide insights into how the model makes its predictions and the contribution of each feature to the final output.

## Dataset
The dataset used in this project is the California Housing Prices dataset, which can be fetched from `sklearn.datasets`. The target variable is the median house value for California districts, and the features include various socio-economic factors.

## Model
A Random Forest Regressor is trained on the dataset to predict the housing prices. The performance of the model is evaluated using the Mean Squared Error (MSE) metric.

## Explainable AI Techniques
### SHAP
SHAP values are computed to understand the contribution of each feature to the model's predictions. The project includes various visualizations to interpret these values:
- Bar Plot
- Summary Plot
- Beeswarm Plot
- Violin Plot
- Waterfall Plot
- Force Plot

### LIME
LIME is used to generate local surrogate models that approximate the predictions of the black-box model locally. The explanations provided by LIME help in understanding the behavior of the model for individual predictions.

## Code Example
Here's a basic example of how to use SHAP in this project:

```python
import shap
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = fetch_california_housing(as_frame=True)
X = dataset['data']
y = dataset['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse}")

# Calculate SHAP values
start_time = time.time()
explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to compute SHAP values: {elapsed_time} seconds")

# Plot SHAP values
shap.plots.bar(shap_values)
shap.summary_plot(shap_values)
shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, plot_type='violin')
shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])
