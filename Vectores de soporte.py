# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:23:38 2020

@author: a779437
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

boston = datasets.load_boston()

#Numero de habitaciones
X_svr = boston.data[:, np.newaxis, 5]

#Datos correspondientes a las etiquetas
y_svr = boston.target

#Separo los datos del "train" en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_svr, y_svr, test_size = 0.2)

# Define el algoritmo, se entrena y se prueba
svr = SVR(kernel = 'linear', C = 1.0, epsilon = 0.2)
svr.fit(X_train, y_train)
Y_pred = svr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_test, color='red')
plt.show()

print(svr.score(X_train, y_train))

