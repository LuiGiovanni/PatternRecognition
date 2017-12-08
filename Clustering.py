import numpy as np
import pandas as pd
import math
import random
import pprint
import time
import os
import sys
from scipy.spatial import distance
from sklearn import cluster
from IPython.display import display
from matplotlib import pyplot as plt

df = pd.read_csv("vgsales_limp_G.csv")

print(df)

df.loc[df['year'] == 0,'year'] = df['year'].mean()
df.loc[df['genre'] == 0,'genre'] = df['genre'].mean()
df.loc[df['NA_Sales'] == 0,'NA_Sales'] = df['NA_Sales'].mean()
df.loc[df['EU_Sales'] == 0,'EU_Sales'] = df['EU_Sales'].mean()
df.loc[df['JP_Sales'] == 0,'JP_Sales'] = df['JP_Sales'].mean()
df.loc[df['Other_Sales'] == 0,'Other_Sales'] = df['Other_Sales'].mean()
df.loc[df['Global_Sales'] == 0,'Global_Sales'] = df['Global_Sales'].mean()

X = df.head(2000)

print(X)

from scipy.spatial import distance
np.set_printoptions(precision=1, suppress=True) # Cortar la impresión de decimales a 1

# Convertir el vector de distancias a una matriz cuadrada
md = distance.squareform(distance.pdist(X, 'euclidean')) 
print(md)

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, 'single')
plt.figure(figsize=(12, 8))
dendrogram(Z, leaf_font_size=14)
plt.show()

Z = linkage(X, 'complete')
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_font_size=14)
plt.show()

Z = linkage(X, 'centroid')
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_font_size=14)
plt.show()

plt.figure(figsize=(12, 5))
dendrogram(
    Z,
    truncate_mode='lastp',  # mostrar sólo los últims p clusters
    p=10,                   
    show_leaf_counts=True,  # mostrar entre paréntesis el número de elementos en cada cluster
    leaf_font_size=14.,
)
plt.show()

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=11, show_leaf_counts=True, leaf_font_size=14.)
plt.show()

md = distance.squareform(distance.pdist(X, 'seuclidean')) 
print(md)

Z = linkage(X, 'single')
plt.figure(figsize=(12, 8))
dendrogram(Z, leaf_font_size=14)
plt.show()

Z = linkage(X, 'complete')
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_font_size=14)
plt.show()

Z = linkage(X, 'centroid')
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_font_size=14)
plt.show()

plt.figure(figsize=(12, 5))
dendrogram(
    Z,
    truncate_mode='lastp',  # mostrar sólo los últims p clusters
    p=10,                   
    show_leaf_counts=True,  # mostrar entre paréntesis el número de elementos en cada cluster
    leaf_font_size=14.,
)
plt.show()

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=11, show_leaf_counts=True, leaf_font_size=14.)
plt.show()