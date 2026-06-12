#pragma once

#include <random>
#include <cmath>

class WeightInitializer {
public:
    virtual float generate(int fan_in, int fan_out) = 0;
    virtual ~WeightInitializer() {}
};

class ZerosInitializer : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        return 0.0f;
    }
};

class RandomUniform : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        return dist(gen);
    }
};

class XavierUniform : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        static std::mt19937 gen(std::random_device{}());
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(gen);
    }
};

class HeUniform : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        static std::mt19937 gen(std::random_device{}());
        float limit = std::sqrt(6.0f / fan_in);
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(gen);
    }
};

class XavierNormal : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        static std::mt19937 gen(std::random_device{}());
        float stddev = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, stddev);
        return dist(gen);
    }
};

class HeNormal : public WeightInitializer {
public:
    float generate(int fan_in, int fan_out) override {
        static std::mt19937 gen(std::random_device{}());
        float stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        return dist(gen);
    }
};
