#include "cuda_forward.cuh"
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

template <typename Activation>
__global__
void forward_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    const float* __restrict__ input,
    float* output,
    float* z_values,
    int input_size,
    int output_size){
        
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float s_input[TILE_SIZE];

    float sum = 0.0f;

    for (int j_start = 0; j_start < input_size; j_start += TILE_SIZE) {
        int input_idx = j_start + tx;
        if (input_idx < input_size) {
            s_input[tx] = input[input_idx];
        } else {
            s_input[tx] = 0.0f;
        }

        __syncthreads();

        if (neuron < output_size) {
            int limit = (input_size - j_start < TILE_SIZE) ? (input_size - j_start) : TILE_SIZE;
            int weight_row_offset = neuron * input_size + j_start;
            for (int j = 0; j < limit; j++) {
                sum += weights[weight_row_offset + j] * s_input[j];
            }
        }

        __syncthreads();
    }

    if (neuron < output_size) {
        sum += biases[neuron];
        if (z_values) {
            z_values[neuron] = sum;
        }
        Activation activation;
        output[neuron] = activation(sum);
    }
}

template <typename Activation>
void cuda_forward_layer(
    const float* d_weights,
    const float* d_biases,
    const float* d_input,
    float* d_output,
    float* d_z, 
    int input_size,
    int output_size){

    int threads = TILE_SIZE;
    int blocks = (output_size + threads - 1) / threads;

    forward_kernel<Activation><<<blocks, threads>>>(d_weights, d_biases, d_input, d_output, d_z, input_size, output_size);
}

// Instanciacion explicita para las funciones de activacion soportadas
template void cuda_forward_layer<Sigmoid>(
    const float* d_weights,
    const float* d_biases,
    const float* d_input,
    float* d_output,
    float* d_z, 
    int input_size,
    int output_size
);

template void cuda_forward_layer<ReLU>(
    const float* d_weights,
    const float* d_biases,
    const float* d_input,
    float* d_output,
    float* d_z, 
    int input_size,
    int output_size
);