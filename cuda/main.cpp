#include <iostream>
#include <vector>
#include <chrono>

#include "mlp.h"
#include "mnist.h"
#include "cargar_simpsons.h"

inline int label_from_one_hot(const std::vector<float>& label) {
    int class_index = 0;
    float max_value = label[0];

    for (int i = 1; i < static_cast<int>(label.size()); i++) {
        if (label[i] > max_value) {
            max_value = label[i];
            class_index = i;
        }
    }

    return class_index;
}

template <typename Activation>
inline float accuracy(MLP<Activation>& mlp, const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y) {
    int correct = 0;

    for (size_t i = 0; i < X.size(); i++) {
        int predicted = mlp.predict_class(X[i]);
        int expected = label_from_one_hot(Y[i]);

        if (predicted == expected) {
            correct++;
        }
    }

    return (float)correct / X.size();
}

void train_mnist() {
    std::cout << "--- Entrenando MNIST en CUDA (Sigmoid) ---" << std::endl;

    std::vector<std::vector<float>> X_train = read_mnist_images("../cpp/data_mnist/train-images-idx3-ubyte", 5000);
    std::vector<std::vector<float>> Y_train = read_mnist_labels("../cpp/data_mnist/train-labels-idx1-ubyte", 5000);
    std::vector<std::vector<float>> X_test =  read_mnist_images("../cpp/data_mnist/t10k-images-idx3-ubyte", 1000);
    std::vector<std::vector<float>> Y_test = read_mnist_labels("../cpp/data_mnist/t10k-labels-idx1-ubyte", 1000);

    if (X_train.empty() || X_test.empty()) {
        std::cerr << "Por favor, verifica la ruta a los datos de MNIST" << std::endl;
        return;
    }

    std::vector<int> layers = {784, 128, 128, 10};
    

    Sigmoid sigmoid;
    XavierUniform xavier;
    MLP<Sigmoid> mlp(layers, 0.1f, &xavier);

    auto start = std::chrono::high_resolution_clock::now();
    mlp.train(X_train, Y_train, 100, 32);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo de entrenamiento: " << diff.count() << " s" << std::endl;

    float acc = accuracy(mlp, X_test, Y_test);
    std::cout << "Accuracy en test: " << acc * 100.0f << "%" << std::endl;
}

int main(){
    train_mnist();
    
    return 0;
}