import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Customer_ID', 'Time_Between_Purchases', 'Churn_Probability']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Vérification des valeurs manquantes
if data[required_columns].isnull().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées dans les colonnes nécessaires.")

# Préparer les données pour l'analyse de survie
data['Churn_Event'] = (data['Churn_Probability'] > 0.5).astype(int)
T = data['Time_Between_Purchases']
E = data['Churn_Event']

# Vérification des valeurs négatives ou nulles dans les durées
if (T <= 0).any():
    raise ValueError("Les durées doivent être strictement positives.")

# Initialiser et ajuster le modèle Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)

# Afficher la courbe de survie
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Courbe de survie Kaplan-Meier')
plt.xlabel('Temps entre les achats (jours)')
plt.ylabel('Probabilité de rétention')
plt.grid(True)
plt.show()

# Sauvegarder les résultats
output_path = 'output/survival_analysis_results.csv'
survival_data = pd.DataFrame({'Time': kmf.survival_function_.index, 'Survival_Probability': kmf.survival_function_['KM_estimate']})
survival_data.to_csv(output_path, index=False)
print(f"Les résultats de l'analyse de survie ont été enregistrés dans : {output_path}")
