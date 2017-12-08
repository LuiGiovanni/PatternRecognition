import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import os
import time
from IPython.display import display

datos = pd.read_csv("vgsales_limp_G.csv")

etiqueta = datos['genre']
datos = datos.drop('genre', 1)

X_train, X_test, y_train, y_test = train_test_split(datos,etiqueta, test_size=0.2)

for i in [0.1,0.5,1,20,100,500,1000]:
    svmLineal = LinearSVC(C=i)
    start_time = time.time()
    svmLineal.fit(X_train, y_train)
    elapsed_time = time.time() - start_time


    preds_Lineal = svmLineal.predict(X_test)
    fails_Lineal = np.sum(y_test != preds_Lineal)
    preds_train_Lineal = svmLineal.predict(X_train)
    fails_train_Lineal = np.sum(y_train != preds_train_Lineal)
    print("SVM Lineal, C= {} (default)\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
       \nPuntos mal clasificados (prueba): {} de {} ({}%)\
       \nAciertos del {}%\nTiempo: {}\n"
      .format(i, fails_train_Lineal, len(y_train), 100*fails_train_Lineal/len(y_train),
              fails_Lineal, len(y_test), 100*fails_Lineal/len(y_test), 
              svmLineal.score(X_test, y_test)*100, elapsed_time))

for i in [0.1,0.5,1,20,100,500,1000]:
    svmRbf = SVC(kernel='rbf', C=i)
    start_time = time.time()
    svmRbf.fit(X_train, y_train)
    elapsed_time = time.time() - start_time


    preds_Rbf = svmRbf.predict(X_test)
    fails_Rbf = np.sum(y_test != preds_Rbf)

    preds_train_Rbf = svmRbf.predict(X_train)
    fails_train_Rbf = np.sum(y_train != preds_train_Rbf)

    print("SVM Rbf, C= {} (default)\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
       \nPuntos mal clasificados (prueba): {} de {} ({}%)\
       \nAciertos del {}%\nTiempo: {}\n"
      .format(i, fails_train_Rbf, len(y_train), 100*fails_train_Rbf/len(y_train),
              fails_Rbf, len(y_test), 100*fails_Rbf/len(y_test), 
              svmRbf.score(X_test, y_test)*100, elapsed_time))


for i in [0.1,0.5,1,20,100,500,1000]:
    svmRbf = SVC(kernel='rbf', C=1000, gamma=i)
    start_time = time.time()
    svmRbf.fit(X_train, y_train)
    elapsed_time = time.time() - start_time


    preds_Rbf = svmRbf.predict(X_test)
    fails_Rbf = np.sum(y_test != preds_Rbf)

    preds_train_Rbf = svmRbf.predict(X_train)
    fails_train_Rbf = np.sum(y_train != preds_train_Rbf)

    print("SVM RBF, C= {} (default)\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
       \nPuntos mal clasificados (prueba): {} de {} ({}%)\
       \nAciertos del {}%\nTiempo: {}\n"
      .format(i, fails_train_Rbf, len(y_train), 100*fails_train_Rbf/len(y_train),
              fails_Rbf, len(y_test), 100*fails_Rbf/len(y_test), 
              svmRbf.score(X_test, y_test)*100, elapsed_time))