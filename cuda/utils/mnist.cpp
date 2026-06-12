#include "mnist.h"

#include <cstdint>
#include <fstream>
#include <iostream>

std::vector<float> one_hot(int label, int num_classes) {
    std::vector<float> encoded(num_classes, 0.0f);
    if (label >= 0 && label < num_classes) {
        encoded[label] = 1.0f;
    }
    return encoded;
}

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::vector<float>> read_mnist_images(const std::string& filename, int max_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir: " << filename << std::endl;
        return {};
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);

    file.read((char*)&rows, sizeof(rows));
    rows = reverse_int(rows);

    file.read((char*)&cols, sizeof(cols));
    cols = reverse_int(cols);

    if (max_images > 0 && max_images < number_of_images) {
        number_of_images = max_images;
    }

    std::vector<std::vector<float>> images(
        number_of_images,
        std::vector<float>(rows * cols)
    );

    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = pixel / 255.0f;
        }
    }

    return images;
}

std::vector<std::vector<float>> read_mnist_labels(const std::string& filename, int max_labels, int num_classes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir: " << filename << std::endl;
        return {};
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    if (max_labels > 0 && max_labels < number_of_labels) {
        number_of_labels = max_labels;
    }

    std::vector<std::vector<float>> labels;
    for (int i = 0; i < number_of_labels; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels.push_back(one_hot((int)label, num_classes));
    }

    return labels;
}
