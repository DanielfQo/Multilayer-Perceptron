#pragma once

#include <string>
#include <vector>

int reverse_int(int i);
std::vector<float> one_hot(int label, int num_classes = 10);

std::vector<std::vector<float>> read_mnist_images(const std::string& filename,int max_images = -1);

std::vector<std::vector<float>> read_mnist_labels(const std::string& filename,int max_labels = -1,int num_classes = 10);