#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <vector>

#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif
#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif

extern "C" void sme_bfmopa_tile(const uint16_t *a_packed,
                                const uint16_t *b_packed,
                                float *output);

asm(R"(
    .arch armv9-a+sme+sve2
    .text
    .align 4
    .global sme_bfmopa_tile
    .type sme_bfmopa_tile, %function
sme_bfmopa_tile:
    smstart
    ptrue p0.h
    ptrue p1.h
    ptrue p2.s
    zero {za}
    ld1h {z0.h}, p1/z, [x0]
    ld1h {z1.h}, p1/z, [x1]
    bfmopa za0.s, p0/m, p0/m, z0.h, z1.h

    cntw x4
    mov w12, #0
    mov x3, #0
1:
    st1w {za0h.s[w12, 0]}, p2, [x2, x3, lsl #2]
    incw x3
    add w12, w12, #1
    cmp w12, w4
    b.lo 1b
    smstop
    ret
    .size sme_bfmopa_tile, .-sme_bfmopa_tile
)");

static uint16_t exact_positive_integer_bf16(int value)
{
    const float fp32 = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &fp32, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
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
    const int rows = svl_bytes / static_cast<int>(sizeof(float));

    // BFMOPA consumes two K values per instruction. Both sources store one
    // adjacent BF16 pair per output row/column. The BFMOPA predicates must be
    // created at halfword granularity so that both members of each pair are active.
    // All values are exact BF16 encodings, so C[r,c] must be r + c + 2.
    std::vector<uint16_t> a_packed(2 * rows);
    std::vector<uint16_t> b_packed(2 * rows);
    std::vector<float> output(rows * rows, 0.0f);
    for (int i = 0; i < rows; ++i) {
        a_packed[2 * i] = exact_positive_integer_bf16(i + 1);
        a_packed[2 * i + 1] = exact_positive_integer_bf16(1);
        b_packed[2 * i] = exact_positive_integer_bf16(1);
        b_packed[2 * i + 1] = exact_positive_integer_bf16(i + 1);
    }

    sme_bfmopa_tile(a_packed.data(), b_packed.data(), output.data());
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < rows; ++c) {
            const float expected = static_cast<float>(r + c + 2);
            std::printf("%7.2f", output[r * rows + c]);
            ok &= std::abs(output[r * rows + c] - expected) == 0.0f;
        }
        std::putchar('\n');
    }
    std::puts(ok ? "PASS: SME BFMOPA packing and ZA store"
                 : "FAIL: SME BFMOPA result mismatch");
    return ok ? 0 : 1;
}
