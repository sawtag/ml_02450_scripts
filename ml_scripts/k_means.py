from sklearn.cluster import KMeans
import numpy as np

# x_data = [5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]
# x = np.array(x_data).reshape(-1, 1)

x = np.array([
[0,2.39,1.73,0.96,3.46,4.07,4.27,5.11],
[2.39,0,1.15,1.76,2.66,5.36,3.54,4.79],
[1.73,1.15,0,1.52,3.01,4.66,3.77,4.90],
[0.96,1.76,1.52,0,2.84,4.25,3.80,4.74],
[3.46,2.66,3.01,2.84,0,4.88,1.41,2.96],
[4.07,5.36,4.66,4.25,4.88,0,5.47,5.16],
[4.27,3.54,3.77,3.80,1.41,5.47,0,2.88],
[5.11,4.79,4.90,4.74,2.96,5.16,2.88,0]
])
"""
k-means without initialization 
"""
kmeans = KMeans(n_clusters=3).fit(x)
print(kmeans.predict(x))

"""
k-means with initialization
"""
# init_points = np.array([5.7, 6.0, 6.2]).reshape(-1, 1)
# kmeans = KMeans(n_clusters=3, init=init_points).fit(x)
# print(kmeans.predict(x))

