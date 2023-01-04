import pandas as pd
import random
import math
from bosdyn.client import math_helpers

K = 3
K_MEANS_ATTEMPTS = 10

def euclidian_distance(obj, sample):
    # determine euclidian distance via 3 dimensions
    return (math.sqrt((obj.x - sample.x)**2 + (obj.y - sample.y)**2 + (obj.z -sample.z)**2))

def k_means_clustering(objects):
    samples = random.choices(objects, k=K)

    smallest_variance = 10000
    best_mean_poses = None
    for i in range(K_MEANS_ATTEMPTS):

        mean_poses, variance = cluster(objects,samples)

        if variance < smallest_variance:
            smallest_variance = variance
            best_mean_poses = mean_poses

    return best_mean_poses

def calculate_variance(clusters, mean_poses):

    for i in range(K):
        x_variance = 0
        y_variance = 0
        z_variance = 0
        cluster = clusters[i]
        mean_pose = mean_poses

        for obj in cluster:
            (obj.x - mean_pose.x)**2
            
    


def cluster(objects, samples):

    converged_mean_poses = []
    converged_clusters = []

    # Run until clusters converge
    while True:

        clusters = []
        mean_poses = []

        for _ in range(K):
            clusters.append([])

        # determine cluster index for each object, and add point to the appropriate cluster
        for o in objects:
            smallest_distance = 100000
            smallest_index = 0
            for i in range(K):
                distance = euclidian_distance(o, samples[i])
                if distance < smallest_distance:
                    smallest_distance = distance
                    smallest_index = i

            clusters[smallest_index].append(o)

        # find mean pose for each cluster
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            x = y = z = 0
            for c in cluster:
                x += c.x
                y += c.y
                z += c.z
            x /= len(cluster)
            y /= len(cluster)
            z /= len(cluster)

            mean_pose = math_helpers.SE3Pose(x, y, z, 0)

            mean_poses.append(mean_pose)

        # determine if the cluster has converged yet
        for i in range(K):
            s = samples[i]
            mean_pose = mean_poses[i]
            if ((s.x != mean_pose.x) or (s.y != mean_pose.y) or (s.z != mean_pose.z)):
                samples = mean_poses
                continue

        converged_mean_poses = mean_poses
        converged_clusters = clusters
        break

    calculate_variance(converged_clusters,converged_mean_poses)

    # return mean_poses if the clusters have converged
    return mean_poses

objects_dictionary = pd.read_pickle(r'object_dictionary.pkl')
objects = list(objects_dictionary.values())

mean_poses = k_means_clustering(objects)

for o in objects:
     print(o.x, o.y, o.z)


print("MEAN FROM CLUSTERS")
for o in mean_poses:
     print(o.x, o.y, o.z)
    

