import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
data = np.array([['', 'Col1', 'Col2'], ['Fila1', 11, 22], ["Fila2", 33, 44]])
print(data)

dataFrame = pd.DataFrame(data = data[1:, 1:], index = data[1:, 0], columns = data[0, 1:])
print(dataFrame)
'''

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print("DataFrame:")
print(df)

a = [1, 2, 3, 4]
b = [20, 20, 30, 40]

plt.plot(a, b, color="blue", linewidth = 3, label = 'linea')
plt.legend()
plt.show()
