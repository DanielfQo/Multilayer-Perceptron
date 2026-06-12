#include "cuda_backward.cuh"
#include <cuda_runtime.h>
#include <math.h>

#ifndef TILE_SIZE
#define TILE_SIZE 256 // 
#endif

template <typename Activation, typename Loss>
__global__
void output_delta_kernel(
    const float* a,
    const float* expected,
    const float* z,
    float* deltas,
    int output_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size){
        Activation act;
        Loss loss_fn;
        float error = loss_fn.derivative(a[i], expected[i]);
        deltas[i] = error * act.derivative(z[i]);
    }
}

template <typename Activation>
__global__
void hidden_delta_kernel(
    const float* weights_next,
    const float* deltas_next,
    const float* z,
    float* deltas,
    int current_size,
    int next_size){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float s_deltas_next[TILE_SIZE];

    float propagated_error = 0.0f;

    for (int k_start = 0; k_start < next_size; k_start += TILE_SIZE) {
        int delta_idx = k_start + tx;
        if (delta_idx < next_size) {
            s_deltas_next[tx] = deltas_next[delta_idx];
        } else {
            s_deltas_next[tx] = 0.0f;
        }

        __syncthreads();

        if (i < current_size) {
            int limit = (next_size - k_start < TILE_SIZE) ? (next_size - k_start) : TILE_SIZE;
            for (int k = 0; k < limit; k++) {
                propagated_error += weights_next[(k_start + k) * current_size + i] * s_deltas_next[k];
            }
        }

        __syncthreads();
    }

    if (i < current_size)
    {
        Activation act;
        deltas[i] = propagated_error * act.derivative(z[i]);
    }
}

__global__
void accumulate_gradients_kernel(
    const float* deltas,
    const float* a_prev,
    float* grad_w,
    float* grad_b,
    int input_size,
    int output_size){

    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron < output_size){
        if (j < input_size){
            int idx = neuron * input_size + j;
            atomicAdd(&grad_w[idx], deltas[neuron] * a_prev[j]);
        }
        if (j == 0) {
            atomicAdd(&grad_b[neuron], deltas[neuron]);
        }
    }
}

template <typename Activation, typename Loss>
void cuda_backward_output(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size){

    int threads = TILE_SIZE;
    int blocks = (output_size + threads - 1) / threads;
    output_delta_kernel<Activation, Loss><<<blocks, threads>>>(d_a, d_expected, d_z, d_deltas, output_size);
}

template <typename Activation>
void cuda_backward_hidden(
    const float* d_weights_next,
    const float* d_deltas_next,
    const float* d_z,
    float* d_deltas,
    int current_size,
    int next_size){

    int threads = TILE_SIZE;
    int blocks = (current_size + threads - 1) / threads;
    hidden_delta_kernel<Activation><<<blocks, threads>>>(d_weights_next, d_deltas_next, d_z, d_deltas, current_size, next_size);
}

void cuda_accumulate_gradients(
    const float* d_deltas,
    const float* d_a_prev,
    float* d_grad_w,
    float* d_grad_b,
    int input_size,
    int output_size){

    dim3 threads(16, 16);
    dim3 blocks((output_size + threads.x - 1) / threads.x, (input_size + threads.y - 1) / threads.y);
    accumulate_gradients_kernel<<<blocks, threads>>>(d_deltas, d_a_prev, d_grad_w, d_grad_b, input_size, output_size);
}

// Instanciacion explicita para las funciones de activacion soportadas
template void cuda_backward_output<Sigmoid, MSE>(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size
);

template void cuda_backward_output<Sigmoid, CrossEntropy>(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size
);

template void cuda_backward_output<ReLU, MSE>(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size
);

template void cuda_backward_output<ReLU, CrossEntropy>(
    const float* d_a,
    const float* d_expected,
    const float* d_z,
    float* d_deltas,
    int output_size
);

template void cuda_backward_hidden<Sigmoid>(
    const float* d_weights_next,
    const float* d_deltas_next,
    const float* d_z,
    float* d_deltas,
    int current_size,
    int next_size
);

template void cuda_backward_hidden<ReLU>(
    const float* d_weights_next,
    const float* d_deltas_next,
    const float* d_z,
    float* d_deltas,
    int current_size,
    int next_size
);
