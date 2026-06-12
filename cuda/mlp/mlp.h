#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_forward.cuh" 

struct Layer{
    int input_size;
    int output_size;

    // cpu
    std::vector<float> weights;
    std::vector<float> biases;

    // gpu
    float* d_weights = nullptr;
    float* d_biases = nullptr;
    float* d_z = nullptr;         
    float* d_a = nullptr;         // activaciones
    float* d_deltas = nullptr;    // deltas
    float* d_grad_w = nullptr;    // gradiente acumulado w
    float* d_grad_b = nullptr;    // gradiente acumulado b
};

#include "initializers.h"
#include "loss_functions.cuh"

template <typename Activation, typename Loss = MSE>
class MLP{
private:
    std::vector<int> topology;
    float learning_rate;
    std::vector<Layer> layers;
    Activation activation_function;

    // activacion capa de entrada
    float* d_input = nullptr;
    float* d_expected = nullptr;

    void allocate_device_memory();
    void free_device_memory();
    void copy_weights_to_device();
    void copy_weights_from_device();

public:
    MLP(const std::vector<int>& topology, float learning_rate = 0.1f, WeightInitializer* init_method = nullptr);
    ~MLP();

    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;

    std::vector<float> forward(const std::vector<float>& input);
    void forward_device(float* d_input_ptr);
    void backward(const std::vector<float>& expected_output);
    void backward_device(float* d_expected_ptr);
    void accumulate_gradients();
    void update_weights(int batch_size);

    void train(
        const std::vector<std::vector<float>>& training_data,
        const std::vector<std::vector<float>>& labels,
        int epochs,
        int batch_size
    );

    std::vector<float> predict(const std::vector<float>& input);
    int predict_class(const std::vector<float>& input);
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
};

#include "mlp.hpp"