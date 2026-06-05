from importlib.resources import path
import math
import random
import numpy as np

class InitializationFunction:
    def initialize(self, shape):
        raise NotImplementedError

class RandomInitialization(InitializationFunction):
    def initialize(self, shape):
        return np.random.uniform(-1.0, 1.0, shape)
    
class XavierInitialization(InitializationFunction):
    def initialize(self, shape):
        input_size = shape[1]
        limit = math.sqrt(6 / input_size)
        return np.random.uniform(-limit, limit, shape)

class HeInitialization(InitializationFunction):
    def initialize(self, shape):
        input_size = shape[1]
        limit = math.sqrt(2 / input_size)
        return np.random.uniform(-limit, limit, shape)

class ActivationFunction:
    def function(self, x):
        raise NotImplementedError
    def derivative(self, x):
        raise NotImplementedError
    
class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        s = self.function(x)
        return s * (1 - s)

class Relu(ActivationFunction):
    def function(self, x):
        return np.maximum(0.0, x)

    def derivative(self, x):
        return (x > 0).astype(float)
    
class LossFunction:
    def function(self, predicted, target):
        raise NotImplementedError
    def derivative(self, predicted, target):
        raise NotImplementedError
    
class MSE(LossFunction):
    def function(self, predicted, target):
        return np.mean((predicted - target) ** 2)
    
    def derivative(self, predicted, target):
        return 2 * (predicted - target) / len(predicted)
    


class NeuronLayer:
    def __init__(self, input_size, output_size, ActivationFunction=Sigmoid, LossFunction=MSE):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = ActivationFunction()
        self.loss = LossFunction()

        self.weights = np.random.uniform(-1.0, 1.0, (output_size, input_size))
        self.biases = np.random.uniform(-1.0, 1.0, output_size)

        self.input = None
        self.z = None
        self.output = None

        self.delta = None

        self.gradients_w = np.zeros_like(self.weights)
        self.gradients_b = np.zeros_like(self.biases)

        

    def forward(self, input_data):

        self.input = input_data

        self.z = np.matmul(self.weights, input_data) + self.biases

        self.output = self.activation.function(self.z)

        return self.output


class MLP:


    def __init__(self, layer_sizes, ActivationFunction=Sigmoid, LossFunction=MSE):
        self.layers = []
        self.output = None

        self.loss_history = []

        self.ActivationFunction = ActivationFunction
        self.loss = LossFunction()

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            num_neurons = layer_sizes[i + 1]

            layer = NeuronLayer(input_size, num_neurons, ActivationFunction, LossFunction)
            self.layers.append(layer)
        


    def forward(self, input_data):
        output = input_data

        for layer in self.layers:
            output = layer.forward(output) # actualiza el output de cada capa y lo pasa a la siguiente capa


        return output
    
    def backward(self, target):

        output_layer = self.layers[-1]

        # dLoss/dOutput
        error = self.loss.derivative(output_layer.output,target)

        # delta salida
        output_layer.delta = ( error *output_layer.activation.derivative(output_layer.z ))

        # capas ocultas
        for l in range(len(self.layers) - 2, -1, -1):

            current_layer = self.layers[l]
            next_layer = self.layers[l + 1]

            propagated_error = (np.matmul(next_layer.weights.T, next_layer.delta) )

            current_layer.delta = (propagated_error *current_layer.activation.derivative( current_layer.z ) )

    def accumulate_gradients(self):

        for layer in self.layers:
            layer.gradients_w += np.outer(layer.delta, layer.input )
            layer.gradients_b += layer.delta
            
    def update_weights(self, batch_size):

        for layer in self.layers:

            layer.weights -= (self.learning_rate *layer.gradients_w /batch_size)

            layer.biases -= ( self.learning_rate *layer.gradients_b /batch_size)

            layer.gradients_w.fill(0.0)
            layer.gradients_b.fill(0.0)

    def train(self,x,y,learning_rate=0.5,epochs=10000,batch_size=32):

        self.learning_rate = learning_rate

        n = len(x)

        for epoch in range(epochs):

            indices = list(range(n))
            random.shuffle(indices)

            total_loss = 0.0

            for start in range(0, n, batch_size):

                end = min(start + batch_size, n)

                batch_indices = indices[start:end]

                # reset gradientes
                for layer in self.layers:
                    layer.gradients_w.fill(0.0)
                    layer.gradients_b.fill(0.0)

                # recorrer batch
                for idx in batch_indices:

                    output = self.forward(x[idx])

                    total_loss += self.loss.function(output,y[idx])

                    self.backward(y[idx])

                    self.accumulate_gradients()

                self.update_weights(len(batch_indices))

            epoch_loss = total_loss / n
            self.loss_history.append(epoch_loss)
            print(
                f"Epoch {epoch+1} "
                f"Loss: {epoch_loss:.6f}"
            )

            with open("loss_history.txt", "w") as f:
                for epoch, loss in enumerate(self.loss_history, start=1):
                    f.write(f"{epoch} {loss}\n")

            
    def predict(self, input_data):
        return self.forward(input_data)

    def predict_class(self, input_data):
        output = self.forward(input_data)[0]
        return 1.0 if output >= 0.5 else 0.0
    
    def params_to_list(self):
        params = []
        for layer in self.layers:
            params.append(layer.weights)
            params.append(layer.biases)
        return params
    
    def load_from_list(self, params):
        idx = 0
        for layer in self.layers:
            layer.weights = params[idx]
            layer.biases = params[idx + 1]
            idx += 2
    
    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.weights
            data[f"B{i}"] = layer.biases
        np.savez(path, **data)
    
    def load(self, path):
        data = np.load(path)

        for i, layer in enumerate(self.layers):
            layer.weights = data[f"W{i}"]
            layer.biases = data[f"B{i}"]