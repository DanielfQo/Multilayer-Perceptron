#include <iostream>
#include <filesystem>
#include <fstream>

#include "mlp.h"
#include "mnist.h"
#include "cargar_simpsons.h"


void train_mnist() {
    ReLU relu;
    Sigmoid sigmoid;
    XavierUniform xavier_uniform;
    HeUniform he_uniform;
    MeanSquaredError mse;

    std::vector<std::vector<float>> X_train = read_mnist_images("data_mnist/train-images-idx3-ubyte", 5000);
    std::vector<std::vector<float>> Y_train = read_mnist_labels("data_mnist/train-labels-idx1-ubyte", 5000);
    std::vector<std::vector<float>> X_test =  read_mnist_images("data_mnist/t10k-images-idx3-ubyte", 1000);
    std::vector<std::vector<float>> Y_test = read_mnist_labels("data_mnist/t10k-labels-idx1-ubyte", 1000);

    std::vector<int> layers = {784, 128, 128, 10};
    MLP mlp(layers, 0.1f, &sigmoid, &xavier_uniform, &mse);
    mlp.train(X_train, Y_train, 10, 32);
    float acc = accuracy(mlp, X_test, Y_test);

    std::cout << "Accuracy en test: " << acc * 100.0f << "%" << std::endl;

}

void train_simpsons(){

    ReLU relu;
    Sigmoid sigmoid;
    XavierUniform xavier_uniform;
    HeUniform he_uniform;
    MeanSquaredError mse;

    Dataset train = loadDataset("train.bin");
    Dataset test  = loadDataset("test.bin");

    std::vector<std::vector<float>> X_train;
    std::vector<std::vector<float>> Y_train;

    std::vector<std::vector<float>> X_test;
    std::vector<std::vector<float>> Y_test;

    int num_classes = 10; 

    splitDataset(train,num_classes,X_train,Y_train);

    splitDataset(test,num_classes, X_test,Y_test);

    std::vector<int> layers = {2352, 1000, 128, 128, 10};
    MLP mlp(layers, 0.3f, &sigmoid, &xavier_uniform, &mse);
    mlp.train(X_train, Y_train, 100, 32);
    float acc = accuracy(mlp, X_test, Y_test);

    std::cout << "Accuracy en test: " << acc * 100.0f << "%" << std::endl;

    mlp.save_model("simpsons.bin");

}

void load_model_simpsons() {
    
    std::string labels[] = {"Bart", "Burns", "Homer", "Krusty", "Lisa", "Marge", "Milhouse", "Moe", "Ned", "Skinner"};

    ReLU relu;
    Sigmoid sigmoid;
    XavierUniform xavier_uniform;
    HeUniform he_uniform;
    MeanSquaredError mse;

    std::vector<int> layers = {2352, 1000, 128, 128, 10};
    MLP mlp(layers, 0.3f, &sigmoid, &xavier_uniform, &mse);
    mlp.load_model("simpsons.bin");

    Dataset test  = loadDataset("test.bin");
    std::vector<std::vector<float>> X_test;
    std::vector<std::vector<float>> Y_test;

    int num_classes = 10;

    int num_pred = 10;
    for (int i = 0; i < num_pred; i++) {
        
        int idx = i;

        splitDataset(test,num_classes, X_test,Y_test);

        int pred = mlp.predict_class(X_test[idx]);

        std::cout << "Prediccion: " << labels[pred];
        std::cout << " | Etiqueta real: " << labels[label_from_one_hot(Y_test[idx])] << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

}

int main() {

    load_model_simpsons();

    return 0;

}