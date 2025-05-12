import numpy as np
from sklearn.linear_model import Perceptron
import os

os.makedirs("result", exist_ok=True)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

def train_and_save(X, y, gate_name):
    model = Perceptron(max_iter=100, eta0=0.05, random_state=42, tol=0.05)
    model.fit(X, y)
    
    with open(f"result/{gate_name}_model.txt", "w") as f:
        f.write(",".join(map(str, model.coef_[0])) + f",{model.intercept_[0]}\n")
    
    print(f"\nModelo {gate_name.upper()}:")
    print("Pesos:", model.coef_[0])
    print("Bias:", model.intercept_[0])
    print("Predicciones:", model.predict(X))

# Entrenar y guardar modelos
train_and_save(X, y_and, "py_and")
train_and_save(X, y_or, "py_or")