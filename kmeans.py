import pandas as pd
import numpy as np
import random
import math
from bosdyn.client import math_helpers

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def convert_to_np(objects):
    arr = []
    for o in objects:
        arr.append([o.x, o.y, o.z])

    return np.array(arr)
        
objects_dictionary = pd.read_pickle(r'object_dictionary.pkl')
objects = list(objects_dictionary.values())
keys = list(objects_dictionary.keys())

y =[]

# This code shouldn't have to be done in the real code base
for k in keys:
    if "door_handle" in k:
        y.append("door_handle")
    else:
        y.append("handle")

X = convert_to_np(objects)


min_score = 10000
best_kmeans = None

# determine best k value
for i in range(2, len(objects)):
    kmeans = KMeans(n_clusters=i, n_init="auto").fit(X)
    score = silhouette_score(X, kmeans.labels_)

    if score < min_score:
        best_kmeans = kmeans
        min_score = score


# map cluster labels into a dictionary with meaningful text
cluster_dictionary = {}
for i,label in enumerate(kmeans.labels_):
    cluster_dictionary[label] = "object_" + str(kmeans.labels_[i]) + "_" + y[i]


print(X)
print(kmeans.labels_)
print([cluster_dictionary[l] for l in kmeans.labels_])

print(cluster_dictionary[kmeans.predict([[0, 0, 0]])[0]])
