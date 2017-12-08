# Inicializar el ambiente
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import cluster # Auxiliar
import os
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

datos = pd.read_csv("vgsales_limp_G.csv")

etiqueta = datos['genre']

datos = datos.drop('genre', 1)

Rangos = datos.max() - datos.min()

for col in datos.columns.values:
    for i, v in datos.iterrows():
        datos[col][i] = datos[col][i]/Rangos[col]

datos.to_csv("vgsales_limpia_NG.csv")
datosx = datos.assign(label = etiqueta)
datosx.to_csv("vgsales_limpia_G.csv")

cut = 4 * len(etiqueta) // 5

train_setB, test_setB = datos[:cut], datos[cut:]
train_targetsetB, test_targetsetB = etiqueta[:cut], etiqueta[cut:]

clfB = BernoulliNB()
clfB.fit(train_setB, train_targetsetB)

predictions_trainB = clfB.predict(train_setB)
fails_trainB = np.sum(train_targetsetB != predictions_trainB)
print("Bernoulli")
print("Puntos mal clasificados en el conjunto de entrenamiento: {} de {} ({}%)\n"
        .format(fails_trainB, len(train_setB), 100*fails_trainB/len(train_setB)))
predictions_testB = clfB.predict(test_setB)
fails_testB = np.sum(test_targetsetB != predictions_testB)
print("Puntos mal clasificados en el conjunto de prueba {} de {} ({}%)\n"
        .format(fails_testB, len(test_setB), 100*fails_testB/len(test_targetsetB)))

train_setM, test_setM = datos[:cut], datos[cut:]
train_targetsetM, test_targetsetM = etiqueta[:cut], etiqueta[cut:]

clfM = MultinomialNB()
clfM.fit(train_setM, train_targetsetM)

print("Multinomial")
predictions_trainM = clfM.predict(train_setM)
fails_trainM = sum(train_targetsetM  != predictions_trainM)
print("Puntos mal clasificados en el conjunto de entrenamiento: {} de {} ({}%)\n"
      .format(fails_trainM, train_setM.shape[0], 100*fails_trainM/train_setM.shape[0]))
predictions_testM = clfM.predict(test_setM)
fails_testM = sum(test_targetsetM  != predictions_testM)
print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
      .format(fails_testM, test_setM.shape[0], 100*fails_testM/test_setM.shape[0]))

train_setG, test_setG = datos[:cut], datos[cut:]
train_targetsetG, test_targetsetG = etiqueta[:cut], etiqueta[cut:]

clfG = GaussianNB()
clfG.fit(train_setG, train_targetsetG)

print("Gauss")
predictions_trainG = clfG.predict(train_setG)
fails_trainG = sum(train_targetsetG  != predictions_trainG)
print("Puntos mal clasificados en el conjunto de entrenamiento: {} de {} ({}%)\n"
      .format(fails_trainG, train_setG.shape[0], 100*fails_trainG/train_setG.shape[0]))
predictions_testG = clfG.predict(test_setG)
fails_testG = sum(test_targetsetG  != predictions_testG)
print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
      .format(fails_testG, test_setG.shape[0], 100*fails_testG/test_setG.shape[0]))