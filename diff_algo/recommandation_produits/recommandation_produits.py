import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.reader import Reader

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Customer_ID', 'Product_ID', 'Average_Order_Value']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {', '.join(missing_columns)}")

# Préparer les données pour Surprise
reader = Reader(rating_scale=(data['Average_Order_Value'].min(), data['Average_Order_Value'].max()))
dataset = Dataset.load_from_df(data[['Customer_ID', 'Product_ID', 'Average_Order_Value']], reader)

# Séparer les données en ensemble d'entraînement et de test
trainset, testset = train_test_split(dataset, test_size=0.25)

# Initialiser et entraîner le modèle SVD
model = SVD()
model.fit(trainset)

# Faire des prédictions sur l'ensemble de test
predictions = model.test(testset)

# Évaluer les performances
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE : {rmse:.2f}")
print(f"MAE : {mae:.2f}")

# Enregistrer les prédictions
output_path = 'output/product_recommendations.csv'
predictions_df = pd.DataFrame([[pred.uid, pred.iid, pred.est] for pred in predictions], columns=['Customer_ID', 'Product_ID', 'Estimated_Rating'])
predictions_df.to_csv(output_path, index=False)
print(f"Les recommandations ont été enregistrées dans : {output_path}")
