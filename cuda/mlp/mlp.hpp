#pragma once

#include "cuda_backward.cuh"
#include "cuda_update.cuh"

template <typename Activation>
void MLP<Activation>::allocate_device_memory() {
    cudaMalloc(&d_input, topology[0] * sizeof(float));
    cudaMalloc(&d_expected, topology.back() * sizeof(float));

    for (auto& layer : layers){
        cudaMalloc(&layer.d_weights, layer.input_size * layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_biases, layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_z, layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_a, layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_deltas, layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_grad_w, layer.input_size * layer.output_size * sizeof(float));
        cudaMalloc(&layer.d_grad_b, layer.output_size * sizeof(float));

        cudaMemset(layer.d_grad_w, 0, layer.input_size * layer.output_size * sizeof(float));
        cudaMemset(layer.d_grad_b, 0, layer.output_size * sizeof(float));
    }
}

template <typename Activation>
void MLP<Activation>::free_device_memory() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if (d_expected) {
        cudaFree(d_expected);
        d_expected = nullptr;
    }

    for (auto& layer : layers){
        if (layer.d_weights) { cudaFree(layer.d_weights); layer.d_weights = nullptr; }
        if (layer.d_biases)  { cudaFree(layer.d_biases);  layer.d_biases = nullptr; }
        if (layer.d_z)       { cudaFree(layer.d_z);       layer.d_z = nullptr; }
        if (layer.d_a)       { cudaFree(layer.d_a);       layer.d_a = nullptr; }
        if (layer.d_deltas)  { cudaFree(layer.d_deltas);  layer.d_deltas = nullptr; }
        if (layer.d_grad_w)  { cudaFree(layer.d_grad_w);  layer.d_grad_w = nullptr; }
        if (layer.d_grad_b)  { cudaFree(layer.d_grad_b);  layer.d_grad_b = nullptr; }
    }
}

template <typename Activation>
void MLP<Activation>::copy_weights_to_device() {
    for (auto& layer : layers){
        cudaMemcpy(layer.d_weights, layer.weights.data(), layer.input_size * layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer.d_biases, layer.biases.data(), layer.output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

template <typename Activation>
void MLP<Activation>::copy_weights_from_device() {
    for (auto& layer : layers){
        cudaMemcpy(layer.weights.data(), layer.d_weights, layer.input_size * layer.output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(layer.biases.data(), layer.d_biases, layer.output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

template <typename Activation>
MLP<Activation>::MLP(const std::vector<int>& topology, float learning_rate, WeightInitializer* init_method)
    : topology(topology), learning_rate(learning_rate) {
    
    // Si no se provee inicializador creamos uno por defecto
    ZerosInitializer default_initializer;
    WeightInitializer* initializer = (init_method == nullptr) ? &default_initializer : init_method;
    
    for (size_t i = 0; i < topology.size() - 1; i++){
        Layer layer;
        layer.input_size = topology[i];
        layer.output_size = topology[i + 1];

        layer.weights.resize(layer.input_size * layer.output_size);
        layer.biases.resize(layer.output_size);

        for (auto& w : layer.weights) {
            w = initializer->generate(layer.input_size, layer.output_size);
        }

        for (auto& b : layer.biases)
            b = 0.0f;

        layers.push_back(layer);
    }

    allocate_device_memory();
    copy_weights_to_device();
}

template <typename Activation>
MLP<Activation>::~MLP() {
    free_device_memory();
}

template <typename Activation>
std::vector<float> MLP<Activation>::forward(const std::vector<float>& input) {
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* current_input = d_input;

    for (auto& layer : layers){
        cuda_forward_layer<Activation>(
            layer.d_weights,
            layer.d_biases,
            current_input,
            layer.d_a,
            layer.d_z,
            layer.input_size,
            layer.output_size
        );
        current_input = layer.d_a;
    }

    std::vector<float> output(layers.back().output_size);
    cudaMemcpy(output.data(), layers.back().d_a, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}

template <typename Activation>
void MLP<Activation>::backward(const std::vector<float>& expected_output) {
    cudaMemcpy(d_expected, expected_output.data(), expected_output.size() * sizeof(float), cudaMemcpyHostToDevice);

    Layer& output_layer = layers.back();
    cuda_backward_output<Activation>(
        output_layer.d_a,
        d_expected,
        output_layer.d_z,
        output_layer.d_deltas,
        output_layer.output_size
    );

    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--) {
        Layer& current = layers[i];
        Layer& next = layers[i + 1];

        cuda_backward_hidden<Activation>(
            next.d_weights,
            next.d_deltas,
            current.d_z,
            current.d_deltas,
            current.output_size,
            next.output_size
        );
    }
}

template <typename Activation>
void MLP<Activation>::accumulate_gradients() {
    float* current_input = d_input;

    for (size_t i = 0; i < layers.size(); i++){
        Layer& layer = layers[i];
        cuda_accumulate_gradients(
            layer.d_deltas,
            current_input,
            layer.d_grad_w,
            layer.d_grad_b,
            layer.input_size,
            layer.output_size
        );
        current_input = layer.d_a;
    }
}

template <typename Activation>
void MLP<Activation>::update_weights(int batch_size) {
    for (auto& layer : layers){
        int size_w = layer.input_size * layer.output_size;
        int size_b = layer.output_size;

        cuda_update_weights(
            layer.d_weights,
            layer.d_biases,
            layer.d_grad_w,
            layer.d_grad_b,
            size_w,
            size_b,
            learning_rate,
            batch_size
        );
    }
}

template <typename Activation>
void MLP<Activation>::train(
    const std::vector<std::vector<float>>& training_data, const std::vector<std::vector<float>>& labels,
    int epochs,int batch_size){

    if (training_data.size() != labels.size()){
        std::cerr << "Error: El numero de muestras y etiquetas no coincide." << std::endl;
        return;
    }

    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; epoch++){
        float total_loss = 0.0f;
        int current_batch = 0;

        std::vector<int> indices(training_data.size());
        for (size_t i = 0; i < indices.size(); i++){
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < training_data.size(); i++){
            int idx = indices[i];

            std::vector<float> predicted = forward(training_data[idx]);

            float sample_loss = 0.0f;
            for (size_t j = 0; j < predicted.size(); j++){
                float err = predicted[j] - labels[idx][j];
                sample_loss += err * err;
            }
            total_loss += sample_loss / predicted.size();

            backward(labels[idx]);
            accumulate_gradients();

            current_batch++;

            if (current_batch == batch_size){
                update_weights(current_batch);
                current_batch = 0;
            }
        }

        if (current_batch > 0){
            update_weights(current_batch);
        }

        std::cout << "Epoch " << epoch + 1 << " | Loss: " << total_loss / training_data.size() << std::endl;
    }

    copy_weights_from_device();
}

template <typename Activation>
std::vector<float> MLP<Activation>::predict(const std::vector<float>& input) {
    return forward(input);
}

template <typename Activation>
int MLP<Activation>::predict_class(const std::vector<float>& input) {
    std::vector<float> output = forward(input);
    int predicted_class = 0;
    float max_value = output[0];

    for (int i = 1; i < static_cast<int>(output.size()); i++){
        if (output[i] > max_value){
            max_value = output[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

template <typename Activation>
void MLP<Activation>::save_model(const std::string& filename) {
    copy_weights_from_device();

    std::ofstream file(filename, std::ios::binary);
    if (!file){
        throw std::runtime_error("No se pudo abrir el archivo para escritura");
    }

    size_t num_layers = topology.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(topology.data()), num_layers * sizeof(int));

    size_t num_weight_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&num_weight_layers), sizeof(size_t));

    for (size_t l = 0; l < num_weight_layers; l++){
        size_t weight_size = layers[l].weights.size();
        file.write(reinterpret_cast<const char*>(&weight_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(layers[l].weights.data()), weight_size * sizeof(float));

        size_t bias_size = layers[l].biases.size();
        file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(layers[l].biases.data()), bias_size * sizeof(float));
    }

    file.close();
}

template <typename Activation>
void MLP<Activation>::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file){
        throw std::runtime_error("No se pudo abrir el archivo para lectura");
    }

    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));

    std::vector<int> loaded_layers(num_layers);
    file.read(reinterpret_cast<char*>(loaded_layers.data()), num_layers * sizeof(int));

    if (loaded_layers != topology){
        throw std::runtime_error("La arquitectura del archivo no coincide con la red actual");
    }

    size_t num_weight_layers;
    file.read(reinterpret_cast<char*>(&num_weight_layers), sizeof(size_t));

    if (num_weight_layers != layers.size()){
        throw std::runtime_error("Numero de capas incompatible");
    }

    for (size_t l = 0; l < num_weight_layers; l++){
        size_t weight_size;
        file.read(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
        if (weight_size != layers[l].weights.size()){
            throw std::runtime_error("Tamaño de pesos incompatible");
        }
        file.read(reinterpret_cast<char*>(layers[l].weights.data()), weight_size * sizeof(float));

        size_t bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(size_t));
        if (bias_size != layers[l].biases.size()){
            throw std::runtime_error("Tamaño de biases incompatible");
        }
        file.read(reinterpret_cast<char*>(layers[l].biases.data()), bias_size * sizeof(float));
    }

    file.close();

    copy_weights_to_device();
}
