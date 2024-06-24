import shap
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Carica il dataset California Housing
dataset = fetch_california_housing(as_frame=True)
X = dataset['data']
y = dataset['target']

# Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestra un modello di regressione (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prevedi sul test set
y_pred = model.predict(X_test)

# Valuta il modello sul test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) sul test set: {mse}")

# Inizia il timer per il calcolo dei valori SHAP
start_time = time.time()

# Calcola i valori SHAP
explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)

# Ferma il timer e calcola il tempo impiegato
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo impiegato per calcolare i valori SHAP: {elapsed_time} secondi")

# Visualizza i valori SHAP
shap.plots.bar(shap_values)
shap.summary_plot(shap_values)
shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, plot_type='violin')

# Visualizza le spiegazioni per una singola istanza
shap.plots.bar(shap_values[0])
shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])
