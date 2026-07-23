#include <arm_bf16.h>
#include <arm_sve.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static float sve_dot(const bfloat16_t *a, const bfloat16_t *b, size_t count)
{
    svfloat32_t acc = svdup_f32(0.0f);
    size_t offset = 0;
    while (offset < count) {
        svbool_t pg = svwhilelt_b16(offset, count);
        svbfloat16_t va = svld1_bf16(pg, a + offset);
        svbfloat16_t vb = svld1_bf16(pg, b + offset);
        acc = svbfdot_f32(acc, va, vb);
        offset += svcnth();
    }
    return svaddv_f32(svptrue_b32(), acc);
}

int main()
{
    constexpr size_t kDim = 128;
    std::vector<bfloat16_t> a(kDim);
    std::vector<bfloat16_t> b(kDim);
    float reference = 0.0f;

    for (size_t i = 0; i < kDim; ++i) {
        const float af = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
        const float bf = static_cast<float>(static_cast<int>(i % 13) - 6) / 4.0f;
        a[i] = vcvth_bf16_f32(af);
        b[i] = vcvth_bf16_f32(bf);
        reference += vcvtah_f32_bf16(a[i]) * vcvtah_f32_bf16(b[i]);
    }

    const float actual = sve_dot(a.data(), b.data(), kDim);
    const float error = std::abs(actual - reference);
    std::printf("SVE VL: %zu bits, BF16 lanes: %zu\n", svcntb() * 8, svcnth());
    std::printf("reference=%.9g sve=%.9g abs_error=%.9g\n",
                reference, actual, error);
    if (error > std::max(1e-5f, std::abs(reference) * 1e-5f)) {
        std::puts("FAIL: SVE BF16 dot mismatch");
        return 1;
    }
    std::puts("PASS: SVE BF16 dot");
    return 0;
}
