# Inicializar el ambiente
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import metrics

from matplotlib import pyplot as plt

multishapes = pd.read_csv("vgsales_limp_G.csv").values
plt.plot(multishapes[:, 0], multishapes[:, 1], 'o', 
         markeredgecolor='0', markerfacecolor='0', markersize=5)
plt.show()

multishapes = pd.read_csv("vgsales_NG.csv").values

num_clusters = 11
k_means = cluster.KMeans(n_clusters=num_clusters, init='random')
k_means.fit(multishapes)

fig = plt.figure(figsize=(8, 6))
colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ffff00', '#f6ff00', 
          '#2f800f', '#a221b5', '#21b5ac', '#b1216c']
for k in range(num_clusters):
    my_members = k_means.labels_ == k
    plt.plot(multishapes[my_members, 0], multishapes[my_members, 1], 'o', 
             markeredgecolor='k', markerfacecolor=colors[k], markersize=5)

db = cluster.DBSCAN(eps=0.3, min_samples=10)
db.fit(multishapes)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(db.labels_)
print(unique_labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(unique_labels) - (1 if -1 in db.labels_ else 0)

fig = plt.figure(figsize=(8, 6))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    my_members = db.labels_ == k

    xy = multishapes[my_members & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', 
             markerfacecolor=col, markeredgecolor='k', markersize=11)

    xy = multishapes[my_members & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', 
             markerfacecolor=col, markeredgecolor='k', markersize=6)
    
plt.title('Número estimado de clusters: %d' % n_clusters_)
plt.show()