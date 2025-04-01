import numpy as np 
import scipy.spatial.distance as ssd  
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram  

def hierarchical_clustering(X, num_clusters):
    """
    Perform hierarchical clustering using Complete Linkage.
    
    Args:
        X (numpy.ndarray): Data matrix (n_samples, n_features).
        num_clusters (int): Number of final clusters.

    Returns:
        list: List of final clusters (each cluster is a list of point indices).
    """
    n_samples = X.shape[0]  # Get number of data points

    # Compute the initial distance matrix (pairwise distances between points)
    distances = np.full((n_samples, n_samples), np.inf)  # Fill with infinity initially
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = distances[j, i] = ssd.euclidean(X[i], X[j])  # Compute Euclidean distance

    clusters = [[i] for i in range(n_samples)]  # Start with each point as its own cluster
    linkage_matrix = []  # Store clustering steps

    while len(clusters) > num_clusters:  # Continue until we reach the desired number of clusters
        min_dist = np.inf  # Initialize minimum distance
        merge_indices = None  # Store indices of clusters to merge

        # Find the two closest clusters using complete linkage distance
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = complete_linkage_distance(clusters[i], clusters[j], distances)
                if dist < min_dist:
                    min_dist = dist
                    merge_indices = (i, j)

        i, j = merge_indices  # Get indices of clusters to merge
        new_cluster = clusters[i] + clusters[j]  # Merge the two clusters
        linkage_matrix.append([clusters[i][0], clusters[j][0], min_dist, len(new_cluster)])  # Store merge info

        # Remove merged clusters and add the new cluster
        clusters.append(new_cluster)
        del clusters[max(i, j)]
        del clusters[min(i, j)]

        # Update the distance matrix to reflect the new merged cluster
        distances = update_distance_matrix(distances, i, j)

    return clusters, np.array(linkage_matrix)  # Return final clusters and linkage matrix


def complete_linkage_distance(cluster1, cluster2, distances):
    """
    Compute Complete Linkage distance (maximum distance between points in two clusters).
    """
    return max(distances[i, j] for i in cluster1 for j in cluster2)  # Max pairwise distance


def update_distance_matrix(distances, i, j):
    """
    Update the distance matrix after merging clusters i and j.
    """
    new_distances = np.delete(distances, (i, j), axis=0)  # Remove rows of merged clusters
    new_distances = np.delete(new_distances, (i, j), axis=1)  # Remove columns of merged clusters
    
    # Compute new distances using complete linkage (max distance between clusters)
    new_column = np.array([max(distances[k, i], distances[k, j]) for k in range(len(distances)) if k != i and k != j])
    new_column = np.append(new_column, np.inf)  # Distance to itself is infinity
    
    # Append new distances for merged cluster
    new_distances = np.column_stack((new_distances, new_column))  # Add new column
    new_distances = np.vstack((new_distances, new_column.T))  # Add new row

    return new_distances  # Return updated distance matrix

# Generate a random dataset (10 points in 2D)
X = np.random.rand(10, 2)

# Perform hierarchical clustering with Complete Linkage
final_clusters, linkage_matrix = hierarchical_clustering(X, num_clusters=3)

# Print results
print("Final Clusters:", final_clusters)
print("Linkage Matrix:\n", linkage_matrix)

# Plot the dendrogram
plt.figure(figsize=(10, 7))  # Set figure size
plt.title('Hierarchical Clustering with Complete Linkage')  # Title of plot
dendrogram(linkage_matrix)  # Generate dendrogram
plt.xlabel('Sample Index')  # X-axis label
plt.ylabel('Distance')  # Y-axis label
plt.show()  # Display the plot
