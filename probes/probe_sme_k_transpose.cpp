#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <vector>

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif
#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif

constexpr int kTokens = 64;
constexpr int kPairs = 64;
constexpr int kTile = 16;

extern "C" void sme_transpose_k_page(const uint32_t *source,
                                     uint32_t *packed);

asm(R"(
    .arch armv9-a+sme+sve2
    .text
    .align 4
    .global sme_transpose_k_page
    .type sme_transpose_k_page, %function
sme_transpose_k_page:
    sub sp, sp, #64
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    ptrue p0.s
    mov x2, #4
    mov x8, x0
    mov x9, x1

1:
    mov x3, #4
    mov x4, x8
    mov x6, x9
2:
    mov w12, #0
    mov x5, x4
3:
    ld1w {za0h.s[w12, 0]}, p0/z, [x5]
    add x5, x5, #256
    add w12, w12, #1
    cmp w12, #16
    b.lo 3b

    mov w12, #0
4:
    st1w {za0v.s[w12, 0]}, p0, [x6]
    add x6, x6, #64
    add w12, w12, #1
    cmp w12, #16
    b.lo 4b

    add x4, x4, #64
    subs x3, x3, #1
    b.ne 2b

    add x8, x8, #4096
    add x9, x9, #4096
    subs x2, x2, #1
    b.ne 1b
    smstop
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ret
    .size sme_transpose_k_page, .-sme_transpose_k_page
)");

__attribute__((noinline))
static void sve_gather_k_page(const uint32_t *source, uint32_t *packed)
{
    const svbool_t pg = svptrue_b32();
    const svuint32_t offsets = svindex_u32(0, 256);
    for (int tb = 0; tb < 4; ++tb) {
        for (int kp = 0; kp < kPairs; ++kp) {
            const uint32_t *src = source + tb * kTile * kPairs + kp;
            uint32_t *dst = packed + (tb * kPairs + kp) * kTile;
            const svuint32_t values =
                svld1_gather_u32offset_u32(pg, src, offsets);
            svst1_u32(pg, dst, values);
        }
    }
}

template <typename Function>
static double benchmark_ns(Function function, const uint32_t *source,
                           uint32_t *packed, int iterations)
{
    for (int i = 0; i < 1000; ++i) {
        function(source, packed);
        asm volatile("" ::: "memory");
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        function(source, packed);
        // The barrier makes every packed page observable to the compiler.
        // Without it, GCC removes repeated C++ gather calls as dead stores.
        asm volatile("" ::: "memory");
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(stop - start).count() /
           iterations;
}

int main()
{
    const int encoded_vl = prctl(PR_SME_GET_VL);
    if (encoded_vl < 0) {
        std::perror("PR_SME_GET_VL");
        return 2;
    }
    const int svl_bytes = encoded_vl & PR_SME_VL_LEN_MASK;
    std::printf("SME SVL: %d bits\n", svl_bytes * 8);
    if (svl_bytes != 64 || svcntw() != 16) {
        std::printf("FAIL: probe requires 512-bit SVE and SME vectors\n");
        return 2;
    }

    std::vector<uint32_t> source(kTokens * kPairs);
    std::vector<uint32_t> expected(kTokens * kPairs);
    std::vector<uint32_t> actual(kTokens * kPairs, 0xdeadbeefU);
    std::vector<uint32_t> gathered(kTokens * kPairs, 0xcafebabeU);

    for (int token = 0; token < kTokens; ++token) {
        for (int pair = 0; pair < kPairs; ++pair) {
            source[token * kPairs + pair] =
                0xa0000000U | (static_cast<uint32_t>(token) << 8) |
                static_cast<uint32_t>(pair);
            const int tb = token / kTile;
            const int lane = token % kTile;
            expected[(tb * kPairs + pair) * kTile + lane] =
                source[token * kPairs + pair];
        }
    }

    sme_transpose_k_page(source.data(), actual.data());
    sve_gather_k_page(source.data(), gathered.data());

    bool ok = true;
    int errors = 0;
    for (int i = 0; i < kTokens * kPairs; ++i) {
        if (actual[i] != expected[i] || gathered[i] != expected[i]) {
            if (errors < 8) {
                std::printf(
                    "mismatch[%d]: expected=%08x sme=%08x gather=%08x\n",
                    i, expected[i], actual[i], gathered[i]);
            }
            ++errors;
            ok = false;
        }
    }

    std::puts(ok ? "PASS: SME K transpose matches gather packing"
                 : "FAIL: SME K transpose layout mismatch");
    if (!ok) {
        std::printf("total mismatches: %d\n", errors);
        return 1;
    }

    constexpr int kIterations = 200000;
    const double gather_ns = benchmark_ns(
        sve_gather_k_page, source.data(), gathered.data(), kIterations);
    const double sme_ns = benchmark_ns(
        sme_transpose_k_page, source.data(), actual.data(), kIterations);
    std::printf("hot-page gather: %.2f ns/page\n", gather_ns);
    std::printf("hot-page SME:    %.2f ns/page\n", sme_ns);
    std::printf("speedup:         %.3fx\n", gather_ns / sme_ns);
    return 0;
}
