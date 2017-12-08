import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datos = pd.read_csv("vgsales_limp_G.csv")

datos = shuffle(datos.iloc[0:100])

Y = datos.iloc[0:100, 6].values # convertir las etiquetas de clase a valores de clase

# Vector de características
X = datos.iloc[0:100, 0:6].values

# Tasa de aprendizaje
eta=0.01

# Vector de pesos inicial
w = np.zeros(X.shape[1])

# Datos para entrenameinto y prueba
train_data, test_data, train_y, test_y = train_test_split(X, Y, test_size=.5)

# Entrenamiento
print("Vector\tTarget\toutput\terror\tw")
vector = 1
for xi, target in zip(train_data, train_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    error = target - output
    w += eta * error * xi
    if(error != 0):
        print(vector, "\t", target, "\t", output, "\t", error, "\t", w)
    vector += 1
    
print ("Pesos finales: ", w)

# Prueba
errores = 0
for xi, target in zip(test_data, test_y) :
    activation = np.dot(xi, w)
    output = np.where(activation >= 0.0, 1, 0)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_data), 
                                                        errores/len(test_data)*100))

