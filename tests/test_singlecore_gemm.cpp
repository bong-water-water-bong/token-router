// Minimal test: single-core INT8 GEMM using direct BOs
// Proves that single_core xclbin works with engine-style API
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    const char* xclbin_path = argc > 1 ? argv[1] : "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV_singlecore.xclbin";
    const char* insts_path  = argc > 2 ? argv[2] : "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV_singlecore.txt";
    int M=128, K=1024, N=4096, m=32, k=64, n=128;

    printf("Loading xclbin: %s\n", xclbin_path);
    auto device = xrt::device(0);
    auto xc_obj = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xc_obj);
    auto xkernels = xc_obj.get_kernels();
    printf("Kernels: %zu\n", xkernels.size());
    for (auto& xk : xkernels) printf("  kernel: %s\n", xk.get_name().c_str());
    
    auto hc = xrt::hw_context(device, xc_obj.get_uuid());
    auto kernel = xrt::kernel(hc, "MLIR_AIE");

    // Load instructions
    FILE* f = fopen(insts_path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(), 4, instr.size(), f);
    fclose(f);
    printf("Instructions: %zu (%ld bytes)\n", instr.size(), sz);

    // Create BOs — match test.cpp exactly
    auto bo_instr = xrt::bo(device, instr.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(device, (size_t)M * K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, (size_t)K * N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(device, (size_t)M * N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_tmp = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    auto bo_trace = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

    // Fill A with known pattern: A[i,j] = (i*K + j) % 127
    int8_t* bufA = bo_a.map<int8_t*>();
    for (int i = 0; i < M*K; i++) bufA[i] = (i % 127) - 63;

    // Fill B with known pattern: B[j,k] = (j*N + k) % 127  
    int8_t* bufB = bo_b.map<int8_t*>();
    for (int i = 0; i < K*N; i++) bufB[i] = (i % 127) - 63;

    // Zero C
    int32_t* bufC = bo_c.map<int32_t*>();
    memset(bufC, 0, M * N * sizeof(int32_t));

    // Copy instructions
    memcpy(bo_instr.map(), instr.data(), instr.size() * sizeof(int));

    // Sync all to device
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run kernel — same args as test.cpp
    printf("Running kernel...\n");
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c, bo_tmp, bo_trace);
    ert_cmd_state r = run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    float us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    printf("Kernel returned: %d (expect %d = COMPLETED)\n", r, 2);
    printf("Time: %.0f us\n", us);

    // Sync C back
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Compute reference for one element using numpy-like dot product
    // C[i,j] = sum_k A[i,k] * B[k,j] for i in [0,M), j in [0,N), k in [0,K)
    // Test a few positions
    int errors = 0;
    srand(1726250518);
    for (int sample = 0; sample < 100; sample++) {
        int i = rand() % M;
        int j = rand() % N;
        int64_t expected = 0;
        for (int k = 0; k < K; k++) {
            expected += (int64_t)bufA[i*K + k] * (int64_t)bufB[k*N + j];
        }
        int32_t actual = bufC[i*N + j];
        int64_t diff = (expected > actual) ? (expected - actual) : (actual - expected);
        if (diff > 1) { // allow 1 quantization error
            printf("  MISMATCH [%d,%d]: expected=%ld actual=%d diff=%ld\n", i, j, expected, actual, diff);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("\nPASS! All 100 samples match (within 1)\n");
        printf("NPU GFLOPS: %.1f\n", 2.0 * M * K * N / (1000.0 * us));
    } else {
        printf("\nFAILED: %d/100 errors\n", errors);
    }
    return errors ? 1 : 0;
}
