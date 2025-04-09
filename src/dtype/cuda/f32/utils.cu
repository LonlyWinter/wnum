#include <stdio.h>


cudaError_t set_gpu(int dev) {
    int dev_count = 0;
    cudaError_t err_count = cudaGetDeviceCount(&dev_count);
    if (err_count != cudaSuccess) {
        return err_count;
    }
    if (dev >= dev_count) {
        return cudaErrorDevicesUnavailable;
    }
    
    cudaError_t err_set = cudaSetDevice(dev);
    return err_set;
}