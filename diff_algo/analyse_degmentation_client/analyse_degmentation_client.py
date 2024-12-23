import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger les données
file_path = 'sales_and_customer_insights.csv'
data = pd.read_csv(file_path)

# Vérifier les colonnes nécessaires
required_columns = ['Purchase_Frequency', 'Average_Order_Value', 'Churn_Probability', 'Lifetime_Value']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"La colonne {col} est manquante dans le dataset.")

# Sélection des variables pour la segmentation client
features = data[required_columns]

# Normalisation des données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Déterminer le nombre optimal de clusters avec la méthode du coude
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Méthode du Coude pour déterminer k')
plt.xlabel('Nombre de Clusters')
plt.ylabel('WCSS')
plt.show()

# Appliquer K-Means avec le nombre optimal de clusters (à ajuster après analyse du graphique)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Ajouter les clusters au dataset
data['Cluster'] = clusters

# Visualiser la répartition des clusters selon deux dimensions principales
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Segmentation Client par K-Means')
plt.xlabel('Fréquence d\'Achat (normalisée)')
plt.ylabel('Valeur Moyenne de Commande (normalisée)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Afficher la taille de chaque cluster
cluster_counts = data['Cluster'].value_counts()
print('Taille de chaque cluster :')
print(cluster_counts)

# Enregistrer les résultats
output_path = 'output/customer_segmentation_results.csv'
data.to_csv(output_path, index=False)
print(f"Les résultats ont été enregistrés dans : {output_path}")
