#include <cuda_runtime.h>
#include "cuda_update.cuh"

__global__
void update_weights_kernel(
    float* weights,
    float* biases,
    float* grad_w,
    float* grad_b,
    int size_w,
    int size_b,
    float learning_rate,
    int batch_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Actualizacion de pesos
    if (i < size_w){
        weights[i] -= learning_rate * grad_w[i] / batch_size;
        grad_w[i] = 0.0f; // Limpiar para el proximo batch
    }

    // Actualizacion de biases (usamos el mismo kernel si mapea, o limitamos)
    if (i < size_b){
        biases[i] -= learning_rate * grad_b[i] / batch_size;
        grad_b[i] = 0.0f; // Limpiar para el proximo batch
    }
}

void cuda_update_weights(
    float* d_weights,
    float* d_biases,
    float* d_grad_w,
    float* d_grad_b,
    int size_w,
    int size_b,
    float learning_rate,
    int batch_size){

    int threads = 256;
    int max_size;
    if(size_w > size_b ){
        max_size = size_w;
    }else{
        max_size = size_b;
    }
    int blocks = (max_size + threads - 1) / threads;

    update_weights_kernel<<<blocks, threads>>>(
        d_weights, d_biases, d_grad_w, d_grad_b,
        size_w, size_b, learning_rate, batch_size
    );
}
