import math
import random


class Perceptron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [random.uniform(0.0, 1.0) for _ in range(self.input_size)] #pesos entre 0 y 1
        self.bias = random.uniform(0.0, 1.0)
        self.function = None
        self.learning_rate = None

        self.input = None
        self.output = None


    @staticmethod
    def step_function(x):
        if x >= 0:
            return 1
        else:
            return 0
        
    @staticmethod
    def sigmoid_function(x):
        return 1 / (1 + math.exp(-x))
    
    def set_input(self, input):
        if len(input) != self.input_size:
            raise ValueError("The size of the input does not match the input size of the perceptron")
        self.input = input
    
    def visualize_weights(self):
        print("Weights of the perceptron:")
        for i in range(len(self.weights)):
            print(f"Peso {i+1}: {self.weights[i]:.4f}")


    def forward(self):
        self.output = 0.0

        for i in range(len(self.input)):
            self.output += self.input[i] * self.weights[i]

        self.output += self.bias

        if self.function == 'step':
            self.output = Perceptron.step_function(self.output)
            return self.output
        elif self.function == 'sigmoid':
            self.output = Perceptron.sigmoid_function(self.output)
            return self.output
        else:
            raise ValueError("funcion no valida")
    
    def backward(self, target):
        error = target - self.output
        if self.function == 'sigmoid':
            delta = error * self.output * (1 - self.output)
        else:
            delta = error
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * delta * self.input[i]

        self.bias += self.learning_rate * delta

    def train(self, x, y, learning_rate=0.1, function='step', epochs=100):
        self.learning_rate = learning_rate
        self.function = function

        for epoch in range(epochs):
            total_error = 0

            for i in range(len(x)): # itera sobre cada muestra de entrenamiento
                self.set_input(x[i])

                prediction = self.forward()
                target = y[i][0]
                self.backward(target)

                total_error += abs(target - prediction)

            if epoch % 10 == 0 or epoch < 20:
                print(f"Epoch {epoch}, error total: {total_error}")

    def predict(self, x):
        if self.x is None:
             raise ValueError("The perceptron has not been trained yet")
        self.set_input(x)
        return self.forward()