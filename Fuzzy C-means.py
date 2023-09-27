import numpy as np
import numpy.random
from scipy.spatial import distance

# Function to initialize the dataset
def initialize_dataset():
    dataset = np.array([
        [85937, 1648, 615850, 69.1],
        [31281, 374, 621703, 67.6],
        [516471, 2321, 4140237, 78.6],
        [350459, 2860, 3942435, 80.5],
        [184279, 1152, 1887426, 63],
        [22239, 562, 371876, 62.1],
        [527799, 2431, 4294765, 86.4],
        [207573, 2919, 1871776, 83],
        [127492, 2619, 1871776, 52.1],
        [58131, 601, 453309, 19.2],
        [156963, 3147, 295672, 80.1],
        [362866, 458, 2928654, 84.6],
        [367046, 1709, 1450457, 76.2],
        [160, 4, 111163, 89.6],
        [421565, 3005, 8579510, 71.2],
    ])
    return dataset

# Function to add an extra column of zeros to the dataset
def add_column_of_zeros(dataset):
    return np.append(dataset, np.zeros((dataset.shape[0], 1)), axis=1)

# Function to perform Fuzzy C-Means clustering
def fuzzy_c_means(dataset, k=3, p=2, max_iterations=100):
    n, d = dataset.shape
    
    # Initialize cluster centers with the correct number of features
    cluster_centers = np.zeros((k, dataset.shape[1]))
    
    # Randomly initialize the weight matrix
    weights = np.random.dirichlet(np.ones(k), size=n)
    np.round(weights, 2)

    # Perform the clustering algorithm for a fixed number of iterations
    for iteration in range(max_iterations):

        # Compute centroids for each cluster
        for j in range(k):
            denominator_sum = sum(np.power(weights[:, j], 2))
            sum_mm = 0
            for i in range(n):
                mm = np.multiply(np.power(weights[i, j], p), dataset[i, :])
                sum_mm += mm
            centroid = sum_mm / denominator_sum
            cluster_centers[j] = centroid  # No need to reshape
            
        # Update the weight matrix based on distance
        for i in range(n):
            denominator_sum_next = 0
            for j in range(k):
                denominator_sum_next += np.power(1 / distance.euclidean(cluster_centers[j, 0:d], dataset[i, 0:d]), 1 / (p - 1))
            for j in range(k):
                w = np.power((1 / distance.euclidean(cluster_centers[j, 0:d], dataset[i, 0:d])), 1 / (p - 1)) / denominator_sum_next
                weights[i, j] = w

    return weights, cluster_centers

if __name__ == "__main__":
    # Set the number of clusters (k) and the distance exponent (p)
    k = 3
    p = 2

    # Initialize the dataset
    dataset = initialize_dataset()

    # Add an extra column of zeros to the dataset
    dataset_with_zeros = add_column_of_zeros(dataset)

    # Perform Fuzzy C-Means clustering
    final_weights, final_cluster_centers = fuzzy_c_means(dataset_with_zeros, k, p)

    # Print the final weights and cluster centers
    print("\nThe final weights: \n", np.round(final_weights, 2))
    print(final_cluster_centers)