#include <iostream>
#include <cuda_runtime.h>

int main(){
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Nombre: "<< prop.name<< std::endl;
    std::cout << "SMs: "<< prop.multiProcessorCount<< std::endl;
    std::cout << "Warp Size: "<< prop.warpSize<< std::endl;
    std::cout << "Max Threads/Block: "<< prop.maxThreadsPerBlock<< std::endl;
    std::cout << "Max Threads/SM: "<< prop.maxThreadsPerMultiProcessor<< std::endl;
    std::cout << "Shared Memory/Block: "<< prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Shared Memory/SM: "<< prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "Registers/Block: "<< prop.regsPerBlock<< std::endl;
    std::cout << "Registers/SM: "<< prop.regsPerMultiprocessor<< std::endl;
    std::cout << "Max Blocks/SM: "<< prop.maxBlocksPerMultiProcessor<< std::endl;
    std::cout << "Global Memory: "<< prop.totalGlobalMem / (1024 * 1024)<< " MB"<< std::endl;
    std::cout << "Compute Capability: "<< prop.major<< "."<< prop.minor<< std::endl;

    return 0;
}