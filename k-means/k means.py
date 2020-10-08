import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


dataset = load_breast_cancer()
arr = dataset.data
    
def random_initialize(k, data):
    centroids = data.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def k_means(k, data):
    centroids = random_initialize(k, data)
    for iteration in range(0,100):
        assignment_arr = np.array([np.argmin([np.dot(point-centroid, point-centroid) for centroid in centroids]) for point in data])
        centroids = np.array([data[assignment_arr == c_k].mean(axis=0) for c_k in range(k)])
    return centroids,assignment_arr

def compute_J(data, centroids, assignment_arr):
    J = 0
    for index in range(len(data)):
        J += np.dot(data[index] - centroids[assignment_arr[index]],data[index] - centroids[assignment_arr[index]])
    return J / data.shape[0]

cost = []
for k in range(2,8):
    centroids,assignment = k_means(k,arr)
    cost.append(compute_J(arr,centroids,assignment))

k=[2,3,4,5,6,7]
plt.figure(0)
plt.plot(k,cost,label = 'Distortion J')
plt.plot(k,cost, 'o')
plt.xlabel('K value')
plt.ylabel('Cost')
plt.legend()
plt.savefig('Distortion', dpi=400)
