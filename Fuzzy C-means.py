import numpy as np
import numpy.random
from scipy.spatial import distance

# Set the number of clusters (k) and the distance exponent (p)
k = 3
p = 2

# Define the dataset (X) with features
X = np.array([
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

n = len(X)  # Number of data points
d = len(X[0])  # Number of features

# Add an extra column of zeros to the dataset
addZeros = np.zeros((n, 1))
X = np.append(X, addZeros, axis=1)

# Print information about the dataset
print("The training data: \n", X)
print("\nTotal number of data: ", n)
print("Total number of features: ", d)
print("Total number of Clusters: ", k)

# Create an empty array for cluster centers
C = np.zeros((k, d + 1))

# Randomly initialize the weight matrix
weight = np.random.dirichlet(np.ones(k), size=n)
np.round(weight, 2)

# Perform the clustering algorithm for a fixed number of iterations (100 in this case)
for it in range(100):

    # Compute centroid for each cluster
    for j in range(k):
        denoSum = sum(np.power(weight[:, j], 2))
        
        sumMM = 0
        for i in range(n):
            mm = np.multiply(np.power(weight[i, j], p), X[i, :])
            sumMM += mm
        cc = sumMM / denoSum
        C[j] = np.reshape(cc, d + 1)

    # Update the weight matrix based on distance
    for i in range(n):
        denoSumNext = 0
        for j in range(k):
            denoSumNext += np.power(1 / distance.euclidean(C[j, 0:d], X[i, 0:d]), 1 / (p - 1))
        for j in range(k):
            w = np.power((1 / distance.euclidean(C[j, 0:d], X[i, 0:d])), 1 / (p - 1)) / denoSumNext
            weight[i, j] = w  

# Print the final weights and cluster centers
print("\nThe final weights: \n", np.round(weight, 2))
print(C)