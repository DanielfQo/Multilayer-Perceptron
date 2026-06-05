#pragma once

#include <string>
#include <vector>

#include <vector>

struct Sample
{
    int label;
    std::vector<float> x;
};

struct Dataset
{
    int nSamples;
    int width;
    int height;
    int channels;
    int nFeatures;

    std::vector<Sample> samples;
};

Dataset loadDataset(const std::string& filename);

void splitDataset(
    const Dataset& dataset,
    int num_classes,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y
);

