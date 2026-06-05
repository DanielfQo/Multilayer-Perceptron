from perceptron import Perceptron
from mlp import MLP

x = [[ 0.0, 0.0],
     [ 0.0, 1.0],
     [ 1.0, 0.0],
     [ 1.0, 1.0]]


y_and = [[0.0], 
        [0.0], 
        [0.0], 
        [1.0]]

y_or = [[0.0],
        [1.0],
        [1.0],
        [1.0]]

y_xor = [[0.0],
         [1.0],
         [1.0],
         [0.0]]


modelo = Perceptron(2)
print("Entrenamiento AND")
modelo.train(x, y_and,learning_rate=5,function= 'sigmoid', epochs=100)

print("Predicciones:")
for i in range(len(x)):
    prediction = modelo.predict(x[i])
    print(f"Entrada: {x[i]}, Prediccion: {prediction:.4f}, Valor esperado: {y_and[i][0]:.4f}")

print("Entrenamiento OR")
modelo.train(x, y_or,learning_rate=5,function= 'sigmoid', epochs=100)

print("Predicciones:")
for i in range(len(x)):
    prediction = modelo.predict(x[i])
    print(f"Entrada: {x[i]}, Prediccion: {prediction:.4f}, Valor esperado: {y_or[i][0]:.4f}")

print("Entrenamiento XOR")
modelo.train(x, y_xor,learning_rate=5,function= 'sigmoid', epochs=100)
for i in range(len(x)):
    prediction = modelo.predict(x[i])
    print(f"Entrada: {x[i]}, Prediccion: {prediction:.4f}, Valor esperado: {y_xor[i][0]:.4f}")


modeloMLP = MLP([2, 20, 10, 40 , 100, 1])
print("Entrenamiento MLP XOR")
modeloMLP.train(x, y_xor, learning_rate=0.3, epochs=100)
for i in range(len(x)):
    prediction = modeloMLP.predict(x[i])
    print(f"Entrada: {x[i]}, Prediccion: {prediction[0]:.4f}, Valor esperado: {y_xor[i][0]:.4f}")
