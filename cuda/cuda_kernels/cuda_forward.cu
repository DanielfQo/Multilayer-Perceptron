#include <cuda_runtime.h>
#include "cuda_forward.cuh"

#include <math.h>

__device__
float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

__global__
void forward_kernel_shared(
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    const float* __restrict__ input,
    float* output,
    int input_size,
    int output_size){
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron >= output_size)
        return;

    float sum = 0.0f;

    __shared__ float s_input[TILE_SIZE];

    for (int tile = 0; tile < input_size; tile += TILE_SIZE)
    {
        int idx = tile + threadIdx.x;

        if (idx < input_size)
            s_input[threadIdx.x] = input[idx];
        else
            s_input[threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        {
            int global_j = tile + j;

            if (global_j < input_size)
            {
                sum += weights[neuron * input_size + global_j] * s_input[j];
            }
        }

        __syncthreads();
    }

    sum += biases[neuron];

    output[neuron] = relu(sum);
}

void cuda_forward_layer(
    const float* h_weights,
    const float* h_biases,
    const float* h_input,
    float* h_output,
    int input_size,
    int output_size){
    float *d_w, *d_b, *d_x, *d_y;

    cudaMalloc(&d_w, input_size * output_size * sizeof(float));
    cudaMalloc(&d_b, output_size * sizeof(float));
    cudaMalloc(&d_x, input_size * sizeof(float));
    cudaMalloc(&d_y, output_size * sizeof(float));

    cudaMemcpy(d_w, h_weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = TILE_SIZE;
    int blocks = (output_size + threads - 1) / threads;

    forward_kernel_shared<<<blocks, threads>>>(d_w, d_b, d_x, d_y,input_size, output_size);

    cudaMemcpy(h_output, d_y, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);
}