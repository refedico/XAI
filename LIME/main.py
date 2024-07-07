import lime
import lime.lime_tabular
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset = fetch_california_housing(as_frame=True)
X = dataset['data']
y = dataset['target']

# Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestra un modello di regressione (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) sul test set: {mse}")

# Inizia il timer per il calcolo delle spiegazioni LIME
start_time = time.time()

# Crea un oggetto LimeTabularExplainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=['price'],
    mode='regression'
)

# Scegli un sottoinsieme casuale di istanze dal test set per la spiegazione media
np.random.seed(0) 
sample_size = 100  
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
sample_instances = X_test.iloc[sample_indices]

# Genera le spiegazioni LIME per il campione scelto
exp = explainer.explain_instance(data_rows=sample_instances.values,
                                 predict_fn=model.predict,
                                 num_features=X.shape[1])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo impiegato per calcolare le spiegazioni LIME: {elapsed_time} secondi")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Feature Importance secondo LIME')
exp.as_pyplot_figure(ax=ax)
plt.show()

# Visualizza le spiegazioni in formato testuale
for i in range(sample_size):
    print(f"Istanza {i+1}:")
    print(exp.as_list(instance=i))
    print()
