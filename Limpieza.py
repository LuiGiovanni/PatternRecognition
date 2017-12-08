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

datosL = pd.read_csv("vgsales.csv")

datos = datosL.sample(frac=1).reset_index(drop=True)

datos.loc[datos['NA_Sales'] == 0,'NA_Sales'] = np.nan
datos = datos.dropna()

datos = datos.reset_index(drop=True)

datos.loc[datos['EU_Sales'] == 0,'EU_Sales'] = np.nan
datos = datos.dropna()

datos = datos.reset_index(drop=True)

datos.loc[datos['JP_Sales'] == 0,'JP_Sales'] = np.nan
datos = datos.dropna()

datos = datos.reset_index(drop=True)

datos.loc[datos['Other_Sales'] == 0,'Other_Sales'] = np.nan
datos = datos.dropna()

datos = datos.reset_index(drop=True)

datos.loc[datos['Global_Sales'] == 0,'Global_Sales'] = np.nan
datos = datos.dropna()

datos = datos.reset_index(drop=True)

for i,iv in datos.iterrows():
    if datos['genre'][i] == 1:
        datos.loc[i,'genre'] = 1
    elif datos['genre'][i] == 2:
        datos.loc[i,'genre'] = 2
    elif datos['genre'][i] == 3:
        datos.loc[i,'genre'] = 3
    elif datos['genre'][i] == 4:
        datos.loc[i,'genre'] = 4
    elif datos['genre'][i] == 5:
        datos.loc[i,'genre'] = 5
    elif datos['genre'][i] == 6:
        datos.loc[i,'genre'] = 6
    elif datos['genre'][i] == 7:
        datos.loc[i,'genre'] = 7
    elif datos['genre'][i] == 8:
        datos.loc[i,'genre'] = 8
    elif datos['genre'][i] == 9:
        datos.loc[i,'genre'] = 9
    elif datos['genre'][i] == 10:
        datos.loc[i,'genre'] = 10
    else:
        datos.loc[i,'genre'] = 11

print("datos con algun genero [1 - 11]")
print(datos)

genero = datos['genre']

datos = datos.drop('genre', 1)

print("datos SIN genre")
print(datos)

Rangos = datos.max() - datos.min()

for col in datos.columns.values:
    for i, v in datos.iterrows():
        datos[col][i] = datos[col][i]/Rangos[col]

print("Datos finales")
print(datos)
datosx = datos.assign(genre=genero)
datosx.to_csv("vgsales_G_Lim.csv",index=False)

