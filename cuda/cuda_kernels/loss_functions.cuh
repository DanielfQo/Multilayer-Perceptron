#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#include <cmath>

struct MSE {
    CUDA_HOST_DEVICE
    inline float loss(float pred, float target) const {
        return 0.5f * (pred - target) * (pred - target);
    }

    CUDA_HOST_DEVICE
    inline float derivative(float pred, float target) const {
        return pred - target;
    }
};

struct CrossEntropy {
    CUDA_HOST_DEVICE
    inline float loss(float pred, float target) const {
        return -(target * logf(pred + 1e-15f) + (1.0f - target) * logf(1.0f - pred + 1e-15f));
    }

    CUDA_HOST_DEVICE
    inline float derivative(float pred, float target) const {
        return (pred - target) / (pred * (1.0f - pred) + 1e-15f);
    }
};
