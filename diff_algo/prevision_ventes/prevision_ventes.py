import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Peak_Sales_Date', 'Average_Order_Value']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Préparer les données pour Prophet
data['Peak_Sales_Date'] = pd.to_datetime(data['Peak_Sales_Date'])
prophet_data = data[['Peak_Sales_Date', 'Average_Order_Value']]
prophet_data.columns = ['ds', 'y']

# Vérification des données vides ou nulles
if prophet_data.isnull().sum().any():
    raise ValueError("Des valeurs manquantes ont été détectées dans les données d'entrée.")

# Initialiser et entraîner le modèle Prophet
model = Prophet()
model.fit(prophet_data)

# Créer un dataframe pour les prévisions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Afficher les prévisions
fig1 = model.plot(forecast)
plt.title("Prévision des ventes avec Prophet")
plt.xlabel("Date")
plt.ylabel("Valeur moyenne des commandes")
plt.grid(True)
plt.show()

# Évaluer les performances
merged = pd.merge(prophet_data, forecast[['ds', 'yhat']], on='ds', how='inner')
actual = merged['y']
predicted = merged['yhat']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print(f"MAE : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² : {r2:.2f}")

# Enregistrer les prévisions
output_path = 'output/sales_forecast.csv'
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_path, index=False)
print(f"Les prévisions ont été enregistrées dans : {output_path}")
