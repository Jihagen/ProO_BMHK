from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from data_transformer import DataTransformer

class Clusterer:
    def __init__(self, algorithm='kmeans', n_clusters=None, **kwargs):
        self.algorithm = algorithm
        self.model = None
        self.kwargs = kwargs
        self.n_clusters = n_clusters

    def determine_optimal_clusters(self, data_scaled, max_clusters=10):
        if self.algorithm == 'kmeans':
            inertia = []
            K = range(1, max_clusters + 1)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data_scaled)
                inertia.append(kmeans.inertia_)

            # Plot the elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(K, inertia, 'bx-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal Number of Clusters')
            plt.show()

            # Determine the elbow point
            diff = np.diff(inertia)
            elbow_point = np.argmin(diff[1:]) + 1  # Adding 1 because the diff array is one element shorter
            self.optimal_clusters = elbow_point + 1  # Adjusted to find the correct elbow point

            print(f"Optimal number of clusters determined: {self.optimal_clusters}")
            return self.optimal_clusters
        elif self.algorithm == 'dbscan':
            raise NotImplementedError(f"Optimal cluster determination not implemented for {self.algorithm}")
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    def train(self, data_scaled):
        if self.algorithm == 'kmeans':
            if self.n_clusters is None:
                n_clusters = self.determine_optimal_clusters(data_scaled)
            else:
                n_clusters = self.n_clusters

            self.model = KMeans(n_clusters=n_clusters, random_state=42, **self.kwargs)
            self.model.fit(data_scaled)
        elif self.algorithm == 'dbscan':
            self.model = DBSCAN(**self.kwargs)
            self.model.fit(data_scaled)
        elif self.algorithm == 'nearestneighbors':
            self.model = NearestNeighbors(**self.kwargs)
            self.model.fit(data_scaled)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
        return self.model

    def predict(self, data_scaled):
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call `train` first.")
        if self.algorithm == 'kmeans':
            return self.model.predict(data_scaled)
        elif self.algorithm == 'dbscan':
            return self.model.labels_
        elif self.algorithm == 'nearestneighbors':
            distances, indices = self.model.kneighbors(data_scaled)
            return indices
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    def plot_k_distance(self, data_scaled, k=5):
        if self.algorithm == 'dbscan':
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(data_scaled)
            distances, indices = neighbors_fit.kneighbors(data_scaled)

            # Sort and plot distances
            distances = np.sort(distances[:, k-1], axis=0)
            plt.figure(figsize=(10, 6))
            plt.plot(distances)
            plt.xlabel('Data Points sorted by distance')
            plt.ylabel(f'{k}th Nearest Neighbor Distance')
            plt.title('K-distance Graph for Determining eps')
            plt.show()
        else:
            raise ValueError(f"K-distance plot is only applicable for DBSCAN")
