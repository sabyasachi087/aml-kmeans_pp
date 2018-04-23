#
import pandas as pd
import numpy as np

clusters = np.array([5., 15., 25.])
df = pd.DataFrame({'X':[1, 3, 9, 11, 12, 37, 43, 45, 60], 'C1':"_", 'C2':"_", 'C3':"_"})
points = np.array(df['X'])
cluster_map = {0:list(), 1:list(), 2:list()}
for p in points:
    diff = []
    for c in clusters:
        diff.append(abs(p - c))
    indx = np.argsort(np.array(diff))[0]
    cluster_map[indx].append(p)

for key in cluster_map.keys():
    a = cluster_map.get(key)
    clusters[key] = np.mean(a)

print(cluster_map)

print("Clusters", np.round(clusters, 1))
# Clusters [[ ??.?]
# [ 11.5]
# [ ??.?]]
