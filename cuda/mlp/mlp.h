#pragma once

#include <vector>

struct Layer
{
    int input_size;
    int output_size;

    std::vector<float> weights;
    std::vector<float> biases;
};

class MLP
{
private:
    std::vector<Layer> layers;

public:
    MLP(const std::vector<int>& topology);

    std::vector<float> forward(const std::vector<float>& input);
};