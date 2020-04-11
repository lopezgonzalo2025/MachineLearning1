# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:08:35 2020

@author: a779437
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

boston = datasets.load_boston()

#Numero de habitaciones
X_adr = boston.data[:, np.newaxis, 5]

#Datos correspondientes a las etiquetas
y_adr = boston.target

#Separo los datos del "train" en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_adr, y_adr, test_size = 0.2)

# Define el algoritmo, se entrena y se prueba
adr = DecisionTreeRegressor(max_depth = 18)
adr.fit(X_train, y_train)
Y_pred = adr.predict(X_test)

#Se modifica la forma de los datos para poder graficarlos
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red')
plt.show()


print(adr.score(X_train, y_train))