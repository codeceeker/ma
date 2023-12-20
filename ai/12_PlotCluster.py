import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic clustered data
data, labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Explicitly set the n_init parameter to suppress the warning
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(data)

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot the true clusters
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Predicted Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
