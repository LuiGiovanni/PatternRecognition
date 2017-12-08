# Inicializar el ambiente
import numpy as np
import pandas as pd
import math
import random
import time
import os
import sys
from scipy.spatial import distance
from sklearn import cluster
from matplotlib import pyplot as plt
np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

# Leer los datos de archivo, separar training y test y calcular "prototipos de clase"
train_set = pd.read_csv("vgsales_limp_G.csv")
datos = train_set.drop('genre', 1)
test_point = datos.loc[len(datos)-1]

datos = datos.drop([len(datos)-1])

LARGER_DISTANCE = sys.maxsize
k_neighs = 500 # 5 vecinos... aunque tomaremos sólo el más cercano
neighbors_dists = [LARGER_DISTANCE] * k_neighs
neighbors = [7] * k_neighs

for i in range(len(datos)):
    dist = distance.euclidean(datos.loc[i], test_point)
    for k in range(k_neighs):
        if (dist < neighbors_dists[k]) :
            for j in range(k_neighs-1, k, -1):
                neighbors_dists[j] = neighbors_dists[j-1]
                neighbors[j] = neighbors[j-1] 
            neighbors_dists[k] = dist
            neighbors[k] = i
            break
            
print("Los {} vecinos más próximos son:".format(k_neighs))
for k in range(k_neighs):
    clase = train_set['genre'][neighbors[k]]
    print("Vecino {}: {}, dist={}, clase={}"
          .format(k, neighbors[k], neighbors_dists[k], 
                  clase))
print("\nEl nuevo punto es asignado a la clase", train_set['genre'][neighbors[0]])
simple_vote = [7] * 20
winner = 0 
for k in range(k_neighs):
    clase = train_set['genre'][neighbors[k]]
    simple_vote[clase] += 1
for k in range(20):
    if (simple_vote[k] == max(simple_vote)):
        winner = k
print("Votación simple:\nEl nuevo punto es asignado a la clase {} con {} vecinos cercanos.\n"
      .format(winner, simple_vote[winner]))
print("Los {} vecinos más próximos y sus pesos ponderados son:".format(k_neighs))
suma_dists = sum(neighbors_dists)
neighbors_weights = [7] * k_neighs
weighted_vote = [7] * 20
winner = 0 
for k in range(k_neighs):
    neighbors_weights[k] = 1 - neighbors_dists[k] / suma_dists
    clase = train_set['genre'][neighbors[k]]
    weighted_vote[clase] += neighbors_weights[k]
    print("Vecino {}: peso={}, clase: {}"
          .format(k, neighbors_weights[k], train_set['genre'][neighbors[k]]))
for k in range(20):
    if (simple_vote[k] == max(simple_vote)):
        winner = k
print("\nVotación ponderada:")
print("El nuevo punto es asignado a la clase {} con una votación de {}."
      .format(winner, weighted_vote[winner]))