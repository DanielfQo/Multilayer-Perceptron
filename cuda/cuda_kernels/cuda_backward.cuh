#pragma once

#include "activations.cuh"
#include "loss_functions.cuh"

template <typename Activation, typename Loss>
void cuda_backward_output(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size
);

template <typename Activation>
void cuda_backward_hidden(
    const float* d_weights_next,
    const float* d_deltas_next,
    const float* d_z,
    float* d_deltas,
    int current_size,
    int next_size
);

void cuda_accumulate_gradients(
    const float* d_deltas,
    const float* d_a_prev,
    float* d_grad_w,
    float* d_grad_b,
    int input_size,
    int output_size
);
