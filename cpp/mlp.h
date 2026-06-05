#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

class LossFunction{
public:
    virtual float loss(const std::vector<float>& predicted, const std::vector<float>& expected) = 0;
    virtual ~LossFunction() {}
};

class MeanSquaredError : public LossFunction {
public:
    float loss(const std::vector<float>& predicted, const std::vector<float>& expected) override {
        float total_loss = 0.0f;

        for (size_t i = 0; i < predicted.size(); i++) {
            float error = predicted[i] - expected[i];
            total_loss += error * error;
        }

        return total_loss / predicted.size();
    }
};

class WeightInitializer {
public:
    virtual float generate(int fan_in, int fan_out) = 0;
    virtual ~WeightInitializer() {}
};

class ZerosInitializer : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        return 0.0f;
    }
};

class RandomUniform : public WeightInitializer {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;

public:
    RandomUniform(float min_value = -1.0f, float max_value = 1.0f)
        : gen(std::random_device{}()), dist(min_value, max_value) {}

    float generate(int fan_in, int fan_out) override {
        return dist(gen);
    }
};

class XavierUniform : public WeightInitializer {
private:
    std::mt19937 gen;

public:
    XavierUniform() : gen(std::random_device{}()) {}

    float generate(int fan_in, int fan_out) override {
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(gen);
    }
};

class HeUniform : public WeightInitializer {
private:
    std::mt19937 gen;

public:
    HeUniform() : gen(std::random_device{}()) {}

    float generate(int fan_in, int fan_out) override {
        float limit = std::sqrt(6.0f / fan_in);
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(gen);
    }
};

class XavierNormal : public WeightInitializer {
private:
    std::mt19937 gen;

public:
    XavierNormal() : gen(std::random_device{}()) {}

    float generate(int fan_in, int fan_out) override {
        float stddev = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, stddev);
        return dist(gen);
    }
};

class HeNormal : public WeightInitializer {
private:
    std::mt19937 gen;

public:
    HeNormal() : gen(std::random_device{}()) {}

    float generate(int fan_in, int fan_out) override {
        float stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        return dist(gen);
    }
};

class Function {
public:
    virtual float function(float x) = 0;
    virtual float derivative(float x) = 0;

    virtual ~Function() {}
};

class Sigmoid : public Function {
public:
    float function(float x) override {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float derivative(float x) override {
        float sigmoid_x = function(x);
        return sigmoid_x * (1.0f - sigmoid_x);
    }
};

class ReLU : public Function {
public:
    float function(float x) override {
        return std::max(0.0f, x);
    }

    float derivative(float x) override {
        return x > 0.0 ? 1.0 : 0.0;
    }
};

class MLP {
    private:
        std::vector<int> layers;
        float learning_rate;
        Function* activation_function;
        WeightInitializer* weight_initializer;
        LossFunction* loss_function;

        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> biases;

        std::vector<std::vector<float>> layer_z_values;
        std::vector<std::vector<float>> layer_activations;

        std::vector<std::vector<float>> gradient_weights;
        std::vector<std::vector<float>> gradient_biases;

        std::vector<std::vector<float>> deltas;

        

    public:
        MLP(std::vector<int> layers, float learning_rate, Function* activation_function, WeightInitializer* weight_initializer = nullptr, LossFunction* loss_function = nullptr) {
            this->layers = layers;
            this->learning_rate = learning_rate;
            this->activation_function = activation_function;
            this->loss_function = loss_function;

            if (loss_function == nullptr) {
                this->loss_function = new MeanSquaredError();
            } else {
                this->loss_function = loss_function;
            }

            if (weight_initializer == nullptr) {
                this->weight_initializer = new RandomUniform();
            } else {
                this->weight_initializer = weight_initializer;
            }

            for (size_t i = 0; i < layers.size() - 1; i++) {
                int input_size = layers[i];
                int output_size = layers[i + 1];

                std::vector<float> layer_weights(input_size * output_size);
                std::vector<float> layer_biases(output_size);

                for (float& w : layer_weights) {
                    w = weight_initializer->generate(input_size, output_size);
                }

                for (float& b : layer_biases) {
                    b = 0.0f;
                }

                weights.push_back(layer_weights);
                biases.push_back(layer_biases);
            }


            gradient_weights.resize(weights.size());
            gradient_biases.resize(biases.size());

            for (size_t l = 0; l < weights.size(); l++) {
                gradient_weights[l] = std::vector<float>(weights[l].size(), 0.0f);

                gradient_biases[l] = std::vector<float>(biases[l].size(), 0.0f);
            }

            deltas.resize(layers.size() - 1);
            for(size_t i = 0; i < deltas.size(); i++) {
                deltas[i].resize(layers[i + 1]);
            }
        }

        std::vector<float> forward_propagation(const std::vector<float>& input) {
            layer_z_values.clear();
            layer_activations.clear();

            std::vector<float> current_input = input;

            layer_activations.push_back(input);

            for (size_t layer = 0; layer < weights.size(); layer++) {
                int input_size = layers[layer];
                int output_size = layers[layer + 1];

                std::vector<float> z_values(output_size, 0.0f);
                std::vector<float> output(output_size, 0.0f);

                for (int neuron = 0; neuron < output_size; neuron++) {
                    float sum = 0.0f;

                    for (int j = 0; j < input_size; j++) {
                        int index = neuron * input_size + j;
                        sum += weights[layer][index] * current_input[j];
                    }

                    sum += biases[layer][neuron];

                    z_values[neuron] = sum;
                    output[neuron] = activation_function->function(sum);
                }

                layer_z_values.push_back(z_values);
                layer_activations.push_back(output);

                current_input = output;
            }

            return current_input;
        }

        void backward(const std::vector<float>& expected_output) {

            int num_layers = layers.size() - 1;

            int output_layer = num_layers - 1;

            for(int i = 0; i < layers.back(); i++) {

                float error =
                    layer_activations.back()[i] -
                    expected_output[i];

                deltas[output_layer][i] =
                    error *
                    activation_function->derivative(
                        layer_z_values[output_layer][i]
                    );
            }

            for(int layer = num_layers - 2; layer >= 0; layer--) {

                int current_size = layers[layer + 1];
                int next_size = layers[layer + 2];

                for(int i = 0; i < current_size; i++) {

                    float propagated_error = 0.0f;

                    for(int k = 0; k < next_size; k++) {

                        int idx =
                            k * current_size + i;

                        propagated_error +=
                            weights[layer + 1][idx] *
                            deltas[layer + 1][k];
                    }

                    deltas[layer][i] =
                        propagated_error *
                        activation_function->derivative(
                            layer_z_values[layer][i]
                        );
                }
            }
        }

        void accumulate_gradients() {

            int num_layers = layers.size() - 1;

            for(int layer = 0; layer < num_layers; layer++) {

                int input_size = layers[layer];
                int output_size = layers[layer + 1];

                for(int neuron = 0; neuron < output_size; neuron++) {
                    for(int j = 0; j < input_size; j++) {
                        int idx =neuron * input_size + j;
                        gradient_weights[layer][idx] +=deltas[layer][neuron] *layer_activations[layer][j];
                    }
                    gradient_biases[layer][neuron] +=deltas[layer][neuron];
                }
            }
        }

        void update_weights(int batch_size) {

            for (size_t layer = 0; layer < weights.size(); layer++) {

                for (size_t i = 0; i < weights[layer].size(); i++) {

                    weights[layer][i] -=
                        learning_rate *
                        gradient_weights[layer][i] /
                        batch_size;

                    gradient_weights[layer][i] = 0.0f;
                }

                for (size_t i = 0; i < biases[layer].size(); i++) {

                    biases[layer][i] -=
                        learning_rate *
                        gradient_biases[layer][i] /
                        batch_size;

                    gradient_biases[layer][i] = 0.0f;
                }
            }
        }

        void train(
            const std::vector<std::vector<float>>& training_data,
            const std::vector<std::vector<float>>& labels,
            int epochs, int batch_size) {

            if(training_data.size() != labels.size()) {
                std::cerr << "Error: El numero de muestras y etiquetas no coincide." << std::endl;
                return;
            }

            std::mt19937 rng(std::random_device{}());

            for (int epoch = 0; epoch < epochs; epoch++) {

                float total_loss = 0.0f;
                int current_batch = 0;

                std::vector<int> indices(training_data.size());

                for (size_t i = 0; i < indices.size(); i++) {
                    indices[i] = i;
                }

                std::shuffle(indices.begin(),indices.end(),rng);

                for (size_t i = 0; i < training_data.size(); i++) {

                    int idx = indices[i];

                    std::vector<float> predicted =forward_propagation(training_data[idx]);

                    total_loss +=loss_function->loss(predicted,labels[idx]);

                    backward(labels[idx]);

                    accumulate_gradients();

                    current_batch++;

                    if (current_batch == batch_size) {

                        update_weights(current_batch);

                        current_batch = 0;
                    }
                }

                if (current_batch > 0) {

                    update_weights(current_batch);
                }

                std::cout
                    << "Epoch "
                    << epoch + 1
                    << " | Loss: "
                    << total_loss / training_data.size()
                    << std::endl;
            }
        }

        std::vector<float> predict(const std::vector<float>& input) {
            return forward_propagation(input);
        }

        int predict_class(const std::vector<float>& input) {
            std::vector<float> output = forward_propagation(input);

            int predicted_class = 0;
            float max_value = output[0];

            for (int i = 1; i < static_cast<int>(output.size()); i++) {
                if (output[i] > max_value) {
                    max_value = output[i];
                    predicted_class = i;
                }
            }

            return predicted_class;
        }
        
        ~MLP() {

            delete weight_initializer;
            delete activation_function;
            delete loss_function;

        }

        void save_model(const std::string& filename) {

            std::ofstream file(filename, std::ios::binary);

            if (!file) {
                throw std::runtime_error("No se pudo abrir el archivo para escritura");
            }

            // Arquitectura
            size_t num_layers = layers.size();
            file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));

            file.write(
                reinterpret_cast<const char*>(layers.data()),
                num_layers * sizeof(int)
            );

            // Pesos y biases
            size_t num_weight_layers = weights.size();
            file.write(
                reinterpret_cast<const char*>(&num_weight_layers),
                sizeof(size_t)
            );

            for (size_t l = 0; l < num_weight_layers; l++) {

                size_t weight_size = weights[l].size();

                file.write(
                    reinterpret_cast<const char*>(&weight_size),
                    sizeof(size_t)
                );

                file.write(
                    reinterpret_cast<const char*>(weights[l].data()),
                    weight_size * sizeof(float)
                );

                size_t bias_size = biases[l].size();

                file.write(
                    reinterpret_cast<const char*>(&bias_size),
                    sizeof(size_t)
                );

                file.write(
                    reinterpret_cast<const char*>(biases[l].data()),
                    bias_size * sizeof(float)
                );
            }

            file.close();
        }

        void load_model(const std::string& filename) {

            std::ifstream file(filename, std::ios::binary);

            if (!file) {
                throw std::runtime_error("No se pudo abrir el archivo para lectura");
            }

            size_t num_layers;

            file.read(
                reinterpret_cast<char*>(&num_layers),
                sizeof(size_t)
            );

            std::vector<int> loaded_layers(num_layers);

            file.read(
                reinterpret_cast<char*>(loaded_layers.data()),
                num_layers * sizeof(int)
            );

            if (loaded_layers != layers) {
                throw std::runtime_error(
                    "La arquitectura del archivo no coincide con la red actual"
                );
            }

            size_t num_weight_layers;

            file.read(
                reinterpret_cast<char*>(&num_weight_layers),
                sizeof(size_t)
            );

            if (num_weight_layers != weights.size()) {
                throw std::runtime_error(
                    "Numero de capas incompatible"
                );
            }

            for (size_t l = 0; l < num_weight_layers; l++) {

                size_t weight_size;

                file.read(
                    reinterpret_cast<char*>(&weight_size),
                    sizeof(size_t)
                );

                if (weight_size != weights[l].size()) {
                    throw std::runtime_error(
                        "Tamaño de pesos incompatible"
                    );
                }

                file.read(
                    reinterpret_cast<char*>(weights[l].data()),
                    weight_size * sizeof(float)
                );

                size_t bias_size;

                file.read(
                    reinterpret_cast<char*>(&bias_size),
                    sizeof(size_t)
                );

                if (bias_size != biases[l].size()) {
                    throw std::runtime_error(
                        "Tamaño de biases incompatible"
                    );
                }

                file.read(
                    reinterpret_cast<char*>(biases[l].data()),
                    bias_size * sizeof(float)
                );
            }

            file.close();
        }
};

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

inline float accuracy(MLP& mlp, const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y) {
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