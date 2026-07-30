#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <linux/prctl.h>
#include <omp.h>
#include <sched.h>
#include <sys/prctl.h>
#include <vector>

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif
#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif

extern "C" void sme_bfmopa_peak_2x2(
    const uint16_t *operands, uint64_t iterations, float *output);

asm(R"(
    .arch armv9-a+sme+sve2
    .text
    .align 4
    .global sme_bfmopa_peak_2x2
    .type sme_bfmopa_peak_2x2, %function
sme_bfmopa_peak_2x2:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    ptrue p0.h
    ptrue p1.h
    ptrue p2.s
    ld1h {z0.h}, p1/z, [x0]
    ld1h {z1.h}, p1/z, [x0, #1, mul vl]
    ld1h {z4.h}, p1/z, [x0, #2, mul vl]
    ld1h {z5.h}, p1/z, [x0, #3, mul vl]
    zero {za}
1:
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z0.h, z5.h
    bfmopa za2.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z1.h, z5.h
    subs x1, x1, #1
    b.ne 1b

    mov w12, #0
    mov x3, x2
2:
    add x4, x3, #1024
    add x5, x3, #2048
    add x6, x3, #3072
    st1w {za0h.s[w12, 0]}, p2, [x3]
    st1w {za1h.s[w12, 0]}, p2, [x4]
    st1w {za2h.s[w12, 0]}, p2, [x5]
    st1w {za3h.s[w12, 0]}, p2, [x6]
    add x3, x3, #64
    add w12, w12, #1
    cmp w12, #16
    b.lo 2b
    smstop
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ret
    .size sme_bfmopa_peak_2x2, .-sme_bfmopa_peak_2x2
)");

static uint16_t to_bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static uint64_t env_u64(const char *name, uint64_t fallback)
{
    const char *text = std::getenv(name);
    if (!text || !*text) {
        return fallback;
    }
    return std::strtoull(text, nullptr, 10);
}

int main()
{
    const int encoded_vl = prctl(PR_SME_GET_VL);
    if (encoded_vl < 0) {
        std::perror("PR_SME_GET_VL");
        return 2;
    }
    const int svl_bytes = encoded_vl & PR_SME_VL_LEN_MASK;
    if (svl_bytes != 64) {
        std::printf("FAIL: expected 512-bit SME SVL, got %d bits\n",
                    svl_bytes * 8);
        return 2;
    }

    alignas(64) uint16_t operands[4 * 32];
    const float input_value = 1.0f / 1024.0f;
    std::fill(std::begin(operands), std::end(operands),
              to_bf16(input_value));

    alignas(64) float check[4 * 16 * 16] = {};
    sme_bfmopa_peak_2x2(operands, 1, check);
    const float expected = 2.0f * input_value * input_value;
    float max_error = 0.0f;
    for (float value : check) {
        max_error = std::max(max_error, std::abs(value - expected));
    }
    std::printf("SME SVL: %d bits\n", svl_bytes * 8);
    std::printf("one-step max abs error: %.9g\n", max_error);
    if (max_error != 0.0f) {
        std::puts("FAIL: 2x2 BFMOPA validation failed");
        return 1;
    }
    std::puts("PASS: four ZA tiles match the expected outer products");

    const uint64_t iterations = env_u64("SME_ITERS", 5000000);
    const int runs = static_cast<int>(env_u64("PROBE_RUNS", 7));
    const int max_threads = omp_get_max_threads();
    std::vector<float> outputs(
        static_cast<size_t>(max_threads) * 4 * 16 * 16);
    std::vector<int> cpus(max_threads, -1);
    std::vector<double> elapsed(runs, 0.0);
    double start = 0.0;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        cpus[tid] = sched_getcpu();
        float *thread_output =
            outputs.data() + static_cast<size_t>(tid) * 4 * 16 * 16;
        sme_bfmopa_peak_2x2(operands, 100000, thread_output);

        for (int run = 0; run < runs; ++run) {
#pragma omp barrier
#pragma omp master
            start = omp_get_wtime();
#pragma omp barrier
            sme_bfmopa_peak_2x2(operands, iterations, thread_output);
#pragma omp barrier
#pragma omp master
            elapsed[run] = omp_get_wtime() - start;
        }
    }

    const int threads = omp_get_max_threads();
    std::vector<int> active_cpus(cpus.begin(), cpus.begin() + threads);
    std::sort(active_cpus.begin(), active_cpus.end());
    const int unique_cpus = static_cast<int>(
        std::unique(active_cpus.begin(), active_cpus.end()) -
        active_cpus.begin());
    std::printf("threads=%d unique_cpus=%d cpu_range=%d-%d\n",
                threads, unique_cpus, active_cpus.front(),
                active_cpus[threads - 1]);
    if (unique_cpus != threads) {
        std::puts("WARNING: OpenMP workers are not on unique CPUs");
    }

    const double flops = static_cast<double>(threads) *
        static_cast<double>(iterations) * 4.0 * 1024.0;
    std::vector<double> tflops;
    tflops.reserve(runs);
    for (int run = 0; run < runs; ++run) {
        const double value = flops / elapsed[run] / 1.0e12;
        tflops.push_back(value);
        std::printf("run %d: %.3f ms  %.3f TFLOPS\n",
                    run + 1, elapsed[run] * 1.0e3, value);
    }
    std::sort(tflops.begin(), tflops.end());
    std::printf("SME peak median: %.3f TFLOPS\n",
                tflops[tflops.size() / 2]);
    std::printf("SME peak best:   %.3f TFLOPS\n", tflops.back());
    return 0;
}
