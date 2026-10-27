// Standalone verification of the Q5 simt kernel (launch_q5_simt_r8_c4) on the 9B output-projection
// shape (rows=4096, k=4096, cols=9) with ALL-ONES data.
// All weights decode to +1 * scale(1.0); x is all 1.0 -> every output element must equal 4096.0.
// If the simt kernel is correct, every output must be 4096.0. A K-indexing or N-indexing bug would
// produce a different value.
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/memory.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_launch.h"

#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

int main() {
    const int rows = 4096, k = 4096, cols = 9, padded_k = 4096;
    const int groups_per_row = padded_k / 64;  // 64

    const int64_t x_bytes = (int64_t)cols * k * sizeof(__nv_bfloat16);
    const int64_t code_bytes  = (int64_t)rows * groups_per_row * Q5RowSplitStorage::kCodeBytesPerGroup;
    const int64_t high_bytes  = (int64_t)rows * groups_per_row * Q5RowSplitStorage::kHighBytesPerGroup;
    const int64_t scale_bytes = (int64_t)rows * groups_per_row * Q5RowSplitStorage::kScaleBytesPerGroup;

    std::vector<__nv_bfloat16> hx((size_t)x_bytes / 2, __float2bfloat16(1.0f));
    std::vector<uint8_t>  hcodes((size_t)code_bytes, 0x11);   // low nibble 1 -> q=1
    std::vector<uint8_t>  hhigh((size_t)high_bytes, 0x00);    // high bit 0
    std::vector<uint16_t> hscales((size_t)scale_bytes / 2, 0x3c00);  // fp16 1.0

    __nv_bfloat16 *dx, *dout;
    uint8_t *dcodes, *dhigh; uint16_t *dscales;
    cudaMalloc(&dx, (size_t)x_bytes);
    cudaMalloc(&dcodes, (size_t)code_bytes);
    cudaMalloc(&dhigh, (size_t)high_bytes);
    cudaMalloc(&dscales, (size_t)scale_bytes);
    cudaMalloc(&dout, (size_t)rows * cols * sizeof(__nv_bfloat16));
    cudaMemset(dout, 0x7E, (size_t)rows * cols * sizeof(__nv_bfloat16));  // NaN sentinel
    cudaMemcpy(dx, hx.data(), (size_t)x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dcodes, hcodes.data(), (size_t)code_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dhigh, hhigh.data(), (size_t)high_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dscales, hscales.data(), (size_t)scale_bytes, cudaMemcpyHostToDevice);

    Weight w;
    w.qtype = QType::Q5_G64_FP16;
    w.n = rows; w.k = k; w.padded_shape[1] = padded_k; w.shape[1] = k;
    w.qdata = dcodes;
    w.qhigh = dhigh;
    w.scales = reinterpret_cast<uint8_t*>(dscales);
    Tensor x; x.data = dx; x.ne[0] = k; x.ne[1] = cols;
    Tensor out; out.data = dout; out.ne[0] = rows; out.ne[1] = cols;

    launch_q5_simt_r8_c4(x, w, out, 0);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("simt launch error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<__nv_bfloat16> hout((size_t)rows * cols);
    cudaMemcpy(hout.data(), dout, (size_t)rows * cols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int unwritten = 0, ok = 0; float mn = 1e30f, mx = -1e30f;
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) {
            float v = __bfloat162float(hout[(size_t)c * rows + r]);
            if (v > 1e30f || std::isnan(v)) ++unwritten;  // sentinel
            else { if (v == 4096.0f) ++ok; if (v < mn) mn = v; if (v > mx) mx = v; }
        }
    printf("simt r8c4 (4096x4096x9) all-ones: unwritten=%d ok=%d/%d min=%.3f max=%.3f\n",
           unwritten, ok, rows * cols, mn, mx);
    printf("RESULT: %s\n", (ok == rows * cols && unwritten == 0) ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dcodes); cudaFree(dhigh); cudaFree(dscales); cudaFree(dout);
    return 0;
}
