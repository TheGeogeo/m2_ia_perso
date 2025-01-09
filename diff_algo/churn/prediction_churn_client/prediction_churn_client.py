import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Purchase_Frequency', 'Average_Order_Value', 'Churn_Probability', 'Lifetime_Value', 'Region']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Prétraitement des données
features = data[['Purchase_Frequency', 'Average_Order_Value', 'Lifetime_Value']]
target = (data['Churn_Probability'] > 0.5).astype(int)

# Normalisation des données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)

# Initialiser et entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Précision : {accuracy:.2f}")
print("Matrice de confusion :")
print(conf_matrix)
print("Rapport de classification :")
print(report)

# Enregistrer les prédictions
output_path = 'output/churn_predictions_client.csv'
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv(output_path, index=False)
print(f"Les résultats ont été enregistrés dans : {output_path}")
