# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:36:13 2020

@author: a779437
"""
### MI PRIMER ALGORITMO DE MACHINE LEARNING ###

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#Dataset
boston = datasets.load_boston()

print("Información: ")
print(boston.keys())
print()

print("Caracteristicas: ")
print(boston.DESCR)
print()

print("Cantidad de datos:")
print(boston.data.shape)
print()

print("Nombres de las columnas:")
print(boston.feature_names)
print()

#Numero de habitaciones
X = boston.data[:, np.newaxis, 5]

#Datos correspondientes a las etiquetas
y = boston.target

#Grafica de los datos
plt.scatter(X, y)
plt.xlabel("Número de habitaciones")
plt.ylabel("Valor medio")
plt.show()

#Separo los datos de "train"
# en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo a utilizar
lr = linear_model.LinearRegression()

#Entreno el modelo
lr.fit(X_train, y_train)

#Realizo la prediccion
Y_pred = lr.predict(X_test)

#Grafico de los datos junto al modelo conseguido
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red')
plt.title("Regresion lineal simple")
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()

#Datos del modelo
#  y = ax + b
a = lr.coef_
b = lr.intercept_
print('La ecuación del modelo es:')
print('y = ', a, 'x', b)
print()

#Precision del modelo con la estadistica de R^2
pres = lr.score(X_train, y_train)
print('Precisión del modelo:')
print(pres)

'''
El codigo en limpio serian solo 7 lineas

boston = datasets.load_boston()
X = boston.data[:, np.newaxis, 5]
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
Y_pred = lr.predict(X_test)

'''


## REGRESION LINEAL MULTIPLE ##
'''
Para regresion lineal multiple se deben agregar variables 
independientes
'''
#Numero de habitaciones, año y distancia
X_multiple = boston.data[:, 5:8]
