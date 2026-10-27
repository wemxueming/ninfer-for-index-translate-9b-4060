// Standalone reproduction: Q5 rowsplit mma kernel on the 9B value_z shape (rows=8192, k=4096).
// All weights decode to +1 * scale(1.0); x is all 1.0 -> every output row must equal 4096.0f.
// Checks whether rows >= 4096 (the Z half) are written.
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace ninfer::ops;
using namespace ninfer::ops::detail;

template <class Schedule>
void run_case(const char* name, int rows, int k, int cols, int padded_k) {
    const int64_t x_bytes = (int64_t)cols * k * sizeof(__nv_bfloat16);
    const int groups_per_row = padded_k / 64;
    const int64_t w_rows   = rows;
    const int64_t code_bytes  = w_rows * groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
    const int64_t high_bytes  = w_rows * groups_per_row * Q5RowSplitStorage::kHighBytesPerGroup;
    const int64_t scale_bytes = w_rows * groups_per_row * Q5RowSplitStorage::kScaleBytesPerGroup;

    std::vector<__nv_bfloat16> hx((size_t)cols * k, __float2bfloat16(1.0f));
    std::vector<uint8_t> hcodes((size_t)code_bytes, 0x11);  // low nibble 1 -> q=1
    std::vector<uint8_t> hhigh((size_t)high_bytes, 0x00);   // high bit 0
    std::vector<uint16_t> hscales((size_t)scale_bytes / 2, 0x3c00);  // fp16 1.0

    __nv_bfloat16 *dx, *dout;
    uint8_t *dcodes, *dhigh; uint16_t *dscales;
    cudaMalloc(&dx, (size_t)x_bytes);
    cudaMalloc(&dcodes, (size_t)code_bytes);
    cudaMalloc(&dhigh, (size_t)high_bytes);
    cudaMalloc(&dscales, (size_t)scale_bytes);
    cudaMalloc(&dout, (size_t)rows * cols * sizeof(__nv_bfloat16));
    cudaMemset(dout, 0x7E, (size_t)rows * cols * sizeof(__nv_bfloat16));  // bf16 NaN sentinel
    cudaMemcpy(dx, hx.data(), (size_t)x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dcodes, hcodes.data(), (size_t)code_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dhigh, hhigh.data(), (size_t)high_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dscales, hscales.data(), (size_t)scale_bytes, cudaMemcpyHostToDevice);

    dim3 grid((rows + Schedule::kBlockRows - 1) / Schedule::kBlockRows,
              (cols + Schedule::kBlockCols - 1) / Schedule::kBlockCols, 1);
    q5_rowsplit_gemm_mma_kernel<Schedule, true, Q5MmaEpilogue::Store><<<grid, Schedule::kThreads>>>(
        dx, dcodes, dhigh, reinterpret_cast<const std::uint8_t*>(dscales),
        static_cast<const __nv_bfloat16*>(nullptr), dout, rows, k, cols, padded_k);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%s: launch error %s\n", name, cudaGetErrorString(err)); }

    std::vector<__nv_bfloat16> hout((size_t)rows * cols);
    cudaMemcpy(hout.data(), dout, (size_t)rows * cols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    auto stat = [&](int r0, int r1, const char* tag) {
        int unwritten = 0, zero = 0, ok = 0; float mn = 1e30f, mx = -1e30f;
        for (int r = r0; r < r1; ++r)
            for (int c = 0; c < cols; ++c) {
                float v = __bfloat162float(hout[(int64_t)c * rows + r]);
                if (v > 1e30f || std::isnan(v)) ++unwritten;  // sentinel 0x7E7E ≈ 1.5e34
                else { if (v == 0.0f) ++zero; if (v == 4096.0f || v == 5120.0f) ++ok; if (v < mn) mn = v; if (v > mx) mx = v; }
            }
        printf("%s [%s rows %d..%d): unwritten=%d zero=%d ok=%d/%d min=%.3f max=%.3f\n",
               name, tag, r0, r1, unwritten, zero, ok, (r1 - r0) * cols, mn, mx);
    };
    stat(0, 256, "A(0:256)");
    stat(4096, 4352, "Z(4096:4352)");
    stat(rows - 256, rows, "Z(end)");
    cudaFree(dx); cudaFree(dcodes); cudaFree(dhigh); cudaFree(dscales); cudaFree(dout);
}

int main() {
    printf("CUDA setDevice: %s\n", cudaGetErrorString(cudaSetDevice(0)));
    using C128 = Q5RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q5FragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q5ScaleLoad::Pair32>;
    using C64  = Q5RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q5FragmentPipeline::PingPong,
                                           Cache::ca, Cache::ca, Q5ScaleLoad::Scalar16>;
    run_case<C128>("mma_r64_c128", 8192, 4096, 128, 4096);
    run_case<C64>("mma_r64_c64", 8192, 4096, 128, 4096);
    run_case<C128>("mma_r64_c128-27b-like", 6144, 5120, 128, 5120);
    return 0;
}
