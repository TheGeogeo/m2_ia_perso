import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Purchase_Frequency', 'Average_Order_Value', 'Time_Between_Purchases', 'Lifetime_Value']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Préparer les données
data = data[required_columns]
if data.isnull().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées dans les colonnes nécessaires.")

# Vérifier les valeurs négatives ou nulles
if (data <= 0).any().any():
    raise ValueError("Les colonnes contiennent des valeurs négatives ou nulles qui ne sont pas valides.")

X = data[['Purchase_Frequency', 'Average_Order_Value', 'Time_Between_Purchases']]
y = data['Lifetime_Value']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialiser et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer les performances
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² : {r2:.2f}")

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Valeur réelle')
plt.ylabel('Valeur prédite')
plt.title('Régression Linéaire - Prédiction de la CLV')
plt.grid(True)
plt.show()

# Sauvegarder les résultats
output_path = 'output/clv_predictions.csv'
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv(output_path, index=False)
print(f"Les résultats des prédictions ont été enregistrés dans : {output_path}")
