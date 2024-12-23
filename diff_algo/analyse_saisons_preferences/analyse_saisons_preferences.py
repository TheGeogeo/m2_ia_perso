import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Peak_Sales_Date', 'Average_Order_Value']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Vérifier les valeurs manquantes
if data[required_columns].isnull().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées dans les colonnes nécessaires.")

# Préparer les données pour l'analyse
data['Peak_Sales_Date'] = pd.to_datetime(data['Peak_Sales_Date'])
data = data[['Peak_Sales_Date', 'Average_Order_Value']]
data.set_index('Peak_Sales_Date', inplace=True)
data = data.resample('D').mean().fillna(0)

# Vérifier la taille des données après rééchantillonnage
if len(data) == 0:
    raise ValueError("Les données sont vides après le rééchantillonnage.")

# Appliquer la décomposition STL
stl = STL(data['Average_Order_Value'], seasonal=7, robust=True)
result = stl.fit()

# Tracer les composantes
plt.figure(figsize=(10, 8))
result.plot()
plt.suptitle('Décomposition STL des ventes')
plt.show()

# Sauvegarder les résultats
output_path = 'output/stl_decomposition_results.csv'
decomposition_df = pd.DataFrame({
    'Date': data.index,
    'Trend': result.trend,
    'Seasonal': result.seasonal,
    'Residual': result.resid
})
decomposition_df.to_csv(output_path, index=False)
print(f"Les résultats de la décomposition STL ont été enregistrés dans : {output_path}")
