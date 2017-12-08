import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datos = pd.read_csv("vgsales_limp_G.csv")

# Normalizar los datos
datos_pureAd = (datos_pure - datos_pure.mean()) / datos_pure.std()

# Re-etiquetar las clase de cada vector ejemplo en [-1,1]
datos_classAd = np.where(datos_class == 0, -1, 1)

X_trainPIDAd, X_testPIDAd, y_trainPIDAd, y_testPIDAd = train_test_split(
    datos_pureAd, datos_classAd, test_size=.2)


# Tasa de aprendizaje
eta=0.01

# Número de iteraciones
niter = 200

# Vector de pesos inicial
wPIDAd = np.zeros(X_trainPIDAd.shape[1] + 1)

# Number of misclassifications
errors = []

# Entrenamiento
for i in range(niter):
    output = np.dot(X_trainPIDAd, wPIDAd[1:]) + wPIDAd[0]
    errors = y_trainPIDAd - output
    wPIDAd[1:] += eta * X_trainPIDAd.T.dot(errors)
    wPIDAd[0] += eta * errors.sum()

# Prueba
errores = 0
for xi, target in zip(X_testPIDAd, y_testPIDAd) :
    activation = np.dot(xi, wPIDAd[1:]) + wPIDAd[0]
    output = np.where(activation >= 0.0, 1, -1)
    if (target != output) :
        errores += 1
print("{} vectores mal clasificados de {} ({}%)".format(errores, len(X_testPIDAd), 
                                                        errores/len(X_testPIDAd)*100))