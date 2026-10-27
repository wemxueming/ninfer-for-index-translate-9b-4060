// gpu_probe.cu -- enumerate visible CUDA devices: name, CC, VRAM.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("visible devices = %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        e = cudaGetDeviceProperties(&p, i);
        if (e != cudaSuccess) {
            printf("dev %d: props failed: %s\n", i, cudaGetErrorString(e));
            continue;
        }
        printf("dev %d: %s | CC %d.%d | vram_mib=%zu | sms=%d\n",
               i, p.name, p.major, p.minor,
               (size_t)(p.totalGlobalMem / (1024 * 1024)), p.multiProcessorCount);
    }
    return 0;
}
