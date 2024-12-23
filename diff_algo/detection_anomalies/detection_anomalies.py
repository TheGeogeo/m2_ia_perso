import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Purchase_Frequency', 'Average_Order_Value']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Préparer les données pour la détection d'anomalies
data = data[required_columns]
if data.isnull().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées dans les colonnes nécessaires.")

# Vérifier les valeurs négatives ou nulles
if (data <= 0).any().any():
    raise ValueError("Les colonnes contiennent des valeurs négatives ou nulles qui ne sont pas valides.")

# Normaliser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Appliquer Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(data_scaled)

# Identifier les anomalies
anomalies = data[data['Anomaly'] == -1]

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.scatter(data['Purchase_Frequency'], data['Average_Order_Value'], c=data['Anomaly'], cmap='coolwarm')
plt.title('Détection d\'anomalies avec Isolation Forest')
plt.xlabel('Fréquence des achats')
plt.ylabel('Valeur moyenne des commandes')
plt.grid(True)
plt.show()

# Sauvegarder les résultats
output_path = 'output/isolation_forest_anomalies.csv'
data.to_csv(output_path, index=False)
print(f"Les résultats de la détection d'anomalies ont été enregistrés dans : {output_path}")
