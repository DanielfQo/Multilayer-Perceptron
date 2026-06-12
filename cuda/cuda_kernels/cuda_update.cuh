#pragma once

// Kernel para aplicar SGD en pesos y biases
void cuda_update_weights(
    float* d_weights,
    float* d_biases,
    float* d_grad_w,
    float* d_grad_b,
    int size_w,
    int size_b,
    float learning_rate,
    int batch_size
);
