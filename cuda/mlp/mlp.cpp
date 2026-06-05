#include "mlp.h"

#include <random>

#include "cuda_forward.cuh"

MLP::MLP(const std::vector<int>& topology)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < topology.size() - 1; i++)
    {
        Layer layer;

        layer.input_size = topology[i];
        layer.output_size = topology[i + 1];

        layer.weights.resize(layer.input_size * layer.output_size);
        layer.biases.resize(layer.output_size);

        for (auto& w : layer.weights)
            w = dist(rng);

        for (auto& b : layer.biases)
            b = 0.0f;

        layers.push_back(layer);
    }
}

std::vector<float> MLP::forward(const std::vector<float>& input)
{
    std::vector<float> current = input;

    for (auto& layer : layers)
    {
        std::vector<float> output(layer.output_size);

        cuda_forward_layer(
            layer.weights.data(),
            layer.biases.data(),
            current.data(),
            output.data(),
            layer.input_size,
            layer.output_size
        );

        current = output;
    }

    return current;
}