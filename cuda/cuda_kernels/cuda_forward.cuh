#pragma once

#include "activations.cuh"

template <typename Activation>
void cuda_forward_layer(
    const float* d_weights,
    const float* d_biases,
    const float* d_input,
    float* d_output,
    float* d_z, 
    int input_size,
    int output_size
);