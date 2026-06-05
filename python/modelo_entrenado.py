from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP, MSE, Sigmoid
from cargar_mnist import cargar_mnist


X_train, Y_train, X_test, Y_test = cargar_mnist()

model = MLP([784, 128, 10], Sigmoid, MSE)
model.load("mlp_mnist.npz")

idx = 1500

pred = model.predict(X_test[idx])

print("Clase predicha:", np.argmax(pred))
print("Clase real:", np.argmax(Y_test[idx]))

imagen = X_test[idx].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.imshow(imagen, cmap="gray")
plt.title(
    f"Predicción: {np.argmax(pred)} | Real: {np.argmax(Y_test[idx])}"
)
plt.axis("off")
plt.show()