// gpu_free.cu -- per-device free/total VRAM via the CUDA runtime (no NVML).
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        printf("dev %d: %s | CC %d.%d | free_mib=%zu total_mib=%zu\n",
               i, p.name, p.major, p.minor,
               free_b / (1024 * 1024), total_b / (1024 * 1024));
    }
    return 0;
}
