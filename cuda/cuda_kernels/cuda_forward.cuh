#pragma once

void cuda_forward_layer(
    const float* h_weights,
    const float* h_biases,
    const float* h_input,
    float* h_output,
    int input_size,
    int output_size
);