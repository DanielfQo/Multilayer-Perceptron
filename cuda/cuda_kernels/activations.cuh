#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#include <cmath>

struct Sigmoid {
    CUDA_HOST_DEVICE
    float operator()(float x) const {
#ifdef __CUDA_ARCH__
        return 1.0f / (1.0f + expf(-x));
#else
        return 1.0f / (1.0f + std::exp(-x));
#endif
    }

    CUDA_HOST_DEVICE
    float derivative(float x) const {
        float sig = operator()(x);
        return sig * (1.0f - sig);
    }
};

struct ReLU {
    CUDA_HOST_DEVICE
    float operator()(float x) const {
        return x > 0.0f ? x : 0.0f;
    }

    CUDA_HOST_DEVICE
    float derivative(float x) const {
        return x > 0.0f ? 1.0f : 0.0f;
    }
};
