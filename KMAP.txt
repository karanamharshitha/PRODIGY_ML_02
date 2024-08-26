import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('customer_data.csv')

print(data.head())
print(data.info())

data = data.dropna()

features = data[['TotalPurchaseAmount', 'PurchaseFrequency', 'AverageBasketSize']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k = 3  # Example: Replace with the optimal number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

data['Cluster'] = clusters
print(data.head())

plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='rainbow')
plt.title('Customer Segments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

sns.pairplot(data, hue='Cluster', vars=['TotalPurchaseAmount', 'PurchaseFrequency', 'AverageBasketSize'])
plt.show()

cluster_analysis = data.groupby('Cluster').mean()
print(cluster_analysis)

import joblib
joblib.dump(kmeans, 'kmeans_model.pkl')

data.to_csv('clustered_customers.csv', index=False)