#include "cargar_simpsons.h"

#include <fstream>
#include <stdexcept>

Dataset loadDataset(const std::string& filename)
{
    Dataset dataset;
    std::ifstream file(filename, std::ios::binary);

    if(!file)
        throw std::runtime_error("No se pudo abrir " + filename);

    file.read(reinterpret_cast<char*>(&dataset.nSamples), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.width), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.height), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.channels), sizeof(int));

    dataset.nFeatures = dataset.width * dataset.height * dataset.channels;
    dataset.samples.resize(dataset.nSamples);

    for(int i = 0; i < dataset.nSamples; i++)
    {
        file.read(reinterpret_cast<char*>(&dataset.samples[i].label), sizeof(int));
        dataset.samples[i].x.resize(dataset.nFeatures);
        file.read(reinterpret_cast<char*>(dataset.samples[i].x.data()), dataset.nFeatures * sizeof(float));
    }

    return dataset;
}

void splitDataset(
    const Dataset& dataset,
    int num_classes,
    std::vector<std::vector<float>>& X,
    std::vector<std::vector<float>>& Y
)
{
    X.clear();
    Y.clear();

    for(const auto& sample : dataset.samples)
    {
        X.push_back(sample.x);
        std::vector<float> one_hot(num_classes, 0.0f);
        one_hot[sample.label] = 1.0f;
        Y.push_back(one_hot);
    }
}
