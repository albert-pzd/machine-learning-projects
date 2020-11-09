import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#load dataset
dataset = load_breast_cancer()
arr = dataset.data

#mean normalization and feature scaling
feature_mean = arr.mean(axis = 0)
feature_scale = arr.std(axis = 0)
arr = (arr - feature_mean) / feature_scale

def PCAa(data, k):
    cov_mat = np.cov(data, rowvar = False)
    U, S, V = np.linalg.svd(cov_mat) 
    return U.T[:k]

mat_U = PCAa(arr,2)
z = np.dot(mat_U, arr.T).T

plt.figure()
plt.scatter(z[:,0],z[:,1],s=20)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Dataset k=2')
plt.savefig('PCA',dpi=400)


