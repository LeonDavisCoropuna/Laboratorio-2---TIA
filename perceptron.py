from sklearn.linear_model import Perceptron
import numpy as np

# Datos para compuerta AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # Salida esperada para AND

# Crear el modelo Perceptrón
clf = Perceptron(max_iter=1000, eta0=0.1, random_state=0)

# Entrenamiento
clf.fit(X, y)

# Predicciones
print("Predicciones:")
print(clf.predict(X))

# Mostrar pesos y bias
print("Pesos:", clf.coef_)
print("Bias:", clf.intercept_)
