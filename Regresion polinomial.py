# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 03:35:43 2020

@author: a779437
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

### REGRESION POLINOMIAL ###
### y = a_1 * x_1 + a_2 * x_1^2 + b ### 
## ^2 es el grado del polinomio


#Dataset
boston = datasets.load_boston()

#Numero de habitaciones
X_p = boston.data[:, np.newaxis, 5]

#Datos correspondientes a las etiquetas
y_p = boston.target

#Grafica de los datos
plt.scatter(X_p, y_p)
plt.xlabel("Número de habitaciones")
plt.ylabel("Valor medio")
plt.show()

#Separo los datos del "train" en entrenamiento y prueba
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_p, y_p, test_size = 0.2)

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree = 2)

#Se transforman las caracteristicas existentes
# en caracteristicas de mayor grado
#Se pasa de la formula: y = ax + b
# a la formula polinomial: y = a_1 * x_1 + a_2 * x_1^2 + b
X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli= poli_reg.fit_transform(X_test_p)


#Defino el algoritmo a utilizar
pr = linear_model.LinearRegression()

#Entreno el modelo
pr.fit(X_train_poli, y_train_p)

#Realizo la prediccion
Y_pred_pr = pr.predict(X_test_poli)

#Grafica de los resultados
plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_pr, color='red')
plt.show()

#Valor encontrado para a
print('Valor de a:')
print(pr.coef_)
print()

#Valor encontrado para b
print('Valor de b:')
print(pr.intercept_)
print()

#Precision
print('Precisión:')
print(pr.score(X_train_poli, y_train_p))









