#include "mlp.h"

#include <iostream>
#include <vector>

int main()
{
    MLP mlp({784, 128, 64, 10});

    std::vector<float> x(784, 1.0f);

    auto y = mlp.forward(x);

    std::cout << "Output size: " << y.size() << std::endl;

    for (auto v : y)
        std::cout << v << " ";

    std::cout << std::endl;

    return 0;
}