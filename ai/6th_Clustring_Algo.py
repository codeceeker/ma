import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-Means clustering with explicit setting of n_init
kmeans = KMeans(n_clusters=4, n_init=10)  # You can adjust the value of n_init as needed
kmeans.fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering')
plt.show()
