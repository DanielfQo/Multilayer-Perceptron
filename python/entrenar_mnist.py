from mnist import MNIST
import numpy as np
from mlp import MLP, MSE, Sigmoid

from cargar_mnist import cargar_mnist, evaluate_accuracy


# =====================
# CARGAR DATOS
# =====================

X_train, Y_train, X_test, Y_test = cargar_mnist()

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

model = MLP(
    [784, 128, 10],
    ActivationFunction=Sigmoid,
    LossFunction=MSE
)


model.train(
    X_train,
    Y_train,
    learning_rate=0.3,
    epochs=50,
    batch_size=32
)


acc = evaluate_accuracy(
    model,
    X_test,
    Y_test
)

print(f"Accuracy: {acc:.4f}")


model.save("mlp_mnist.npz")