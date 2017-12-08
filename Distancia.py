import pandas as pd
from scipy.spatial import distance

datos = pd.read_csv("vgsales_NG.csv")

print(datos)

d1 = distance.pdist(datos,'euclidean')
print("\Las distancias euclideanas ({}) para los datos de las ventas son:\n".format(d1.size), d1)
d2 = distance.pdist(datos,'minkowski')
print("\Las distancias de minkowski ({}) para los datos de las ventas son:\n".format(d2.size), d2)
d3 = distance.pdist(datos,'chebyshev')
print("\Las distancias de chebyshev ({}) para los datos de las ventas son:\n".format(d3.size), d3)
d4 = distance.pdist(datos,'seuclidean')
print("\Las distancias de seuclidean ({}) para los datos de las ventas son:\n".format(d4.size), d4)
d5 = distance.pdist(datos,'hamming')
print("\Las distancias de hamming ({}) para los datos de las ventas son:\n".format(d5.size), d5)
d6 = distance.pdist(datos,'cosine')
print("\Las distancias de cosine ({}) para los datos de las ventas son:\n".format(d6.size), d6)

sd1 = 1/(1+d1)
print("\nLas similaridades euclideanas para los datos de ventas son:\n", sd1)
sd2 = 1/(1+d2)
print(sd2)
sd3 = 1/(1+d3)
print(sd3)
sd4 = 1/(1+d4)
print("\nLas similaridaes seuclideanas para los datos de ventas son:\n", sd4)
sd5 = 1/(1+d5)
print(sd5)
sd6 = 1//(1+d6)
print(sd6)

