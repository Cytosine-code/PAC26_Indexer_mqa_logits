#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "indexer_mqa_logits.h"
#include "kupl_mma.h"

using namespace kupl::tensor;

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kTokens = 64;
constexpr int kIterations = 20000;

double now_ns()
{
    return std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pack_q_sme(const bfloat16_t *q, bfloat16_t *packed)
{
    for (int kp = 0; kp < kDim / 2; ++kp) {
        for (int hb = 0; hb < kHeads / 16; ++hb) {
            for (int h = 0; h < 16; ++h) {
                for (int p = 0; p < 2; ++p) {
                    packed[(kp * 4 + hb) * 32 + h * 2 + p] =
                        q[(hb * 16 + h) * kDim + kp * 2 + p];
                }
            }
        }
    }
}

void pack_q_kupl(const bfloat16_t *q, bfloat16_t *packed)
{
    for (int hb = 0; hb < kHeads / 16; ++hb) {
        for (int kp = 0; kp < kDim / 2; ++kp) {
            for (int h = 0; h < 16; ++h) {
                for (int p = 0; p < 2; ++p) {
                    packed[(hb * 64 + kp) * 32 + h * 2 + p] =
                        q[(hb * 16 + h) * kDim + kp * 2 + p];
                }
            }
        }
    }
}

void pack_k(const bfloat16_t *k, bfloat16_t *packed)
{
    for (int kp = 0; kp < kDim / 2; ++kp) {
        for (int t = 0; t < kTokens; ++t) {
            packed[kp * kTokens * 2 + t * 2 + 0] =
                k[t * kDim + kp * 2 + 0];
            packed[kp * kTokens * 2 + t * 2 + 1] =
                k[t * kDim + kp * 2 + 1];
        }
    }
}

inline void kupl_page_scores_m16x4(
    bfloat16_t *packed_q, bfloat16_t *packed_k, float *scores)
{
    auto shape_a = make_shape(Int<16>{}, make_shape(Int<2>{}, Int<64>{}));
    auto shape_b = make_shape(make_shape(Int<2>{}, Int<64>{}), Int<64>{});
    auto shape_c = make_shape(Int<16>{}, Int<64>{});

    auto stride_a = make_stride(Int<2>{}, make_stride(Int<1>{}, Int<32>{}));
    auto stride_b = make_stride(make_stride(Int<1>{}, Int<128>{}), Int<2>{});
    auto stride_c = make_stride(Int<64>{}, Int<1>{});

    auto layout_a = make_layout(shape_a, stride_a);
    auto layout_b = make_layout(shape_b, stride_b);
    auto layout_c = make_layout(shape_c, stride_c);
    auto tiled_mma = make_tiled_mma(
        Ops<KP36_16x64x2_BF16BF16F32>{},
        make_shape(Int<1>{}, Int<1>{}, Int<64>{}));
    auto tiled_store = make_tiled_store(
        Ops<KP36_16x64_F32_STORE>{}, make_shape(Int<1>{}, Int<1>{}));

    auto tensor_b = make_tensor(packed_k, layout_b);
    for (int hb = 0; hb < 4; ++hb) {
        auto tensor_a = make_tensor(packed_q + hb * 16 * kDim, layout_a);
        auto tensor_c = make_tensor(scores + hb * 16 * 64, layout_c);
        mma(tiled_mma, tensor_c, tensor_a, tensor_b, tensor_c);
        store(tiled_store, tensor_c);
    }
}

} // namespace

int main()
{
#ifndef __aarch64__
    std::puts("SKIP: AArch64 with SME and KUPL is required");
    return 0;
#else
    std::vector<bfloat16_t> q(kHeads * kDim);
    std::vector<bfloat16_t> k(kTokens * kDim);
    std::vector<bfloat16_t> sme_packed_q(q.size());
    std::vector<bfloat16_t> kupl_packed_q(q.size());
    std::vector<bfloat16_t> sme_packed_k(k.size());
    std::vector<bfloat16_t> kupl_packed_k(k.size());
    std::vector<float> reference(kHeads * kTokens);
    std::vector<float> sme_scores(reference.size());
    std::vector<float> kupl_m16_scores(reference.size());

    for (int h = 0; h < kHeads; ++h) {
        for (int d = 0; d < kDim; ++d) {
            q[h * kDim + d] = static_cast<bfloat16_t>((h + 3 * d) % 7 - 3);
        }
    }
    for (int t = 0; t < kTokens; ++t) {
        for (int d = 0; d < kDim; ++d) {
            k[t * kDim + d] = static_cast<bfloat16_t>((2 * t + d) % 5 - 2);
        }
    }
    pack_q_sme(q.data(), sme_packed_q.data());
    pack_q_kupl(q.data(), kupl_packed_q.data());
    pac_sme_pack_k_page(k.data(), sme_packed_k.data());
    pack_k(k.data(), kupl_packed_k.data());

    for (int h = 0; h < kHeads; ++h) {
        for (int t = 0; t < kTokens; ++t) {
            float sum = 0.0f;
            for (int d = 0; d < kDim; ++d) {
                sum += static_cast<float>(q[h * kDim + d]) *
                    static_cast<float>(k[t * kDim + d]);
            }
            reference[h * kTokens + t] = sum;
        }
    }

    pac_sme_page_scores_2x2(
        sme_packed_q.data(), sme_packed_k.data(), sme_scores.data());
    std::fill(kupl_m16_scores.begin(), kupl_m16_scores.end(), 0.0f);
    kupl_page_scores_m16x4(
        kupl_packed_q.data(), kupl_packed_k.data(), kupl_m16_scores.data());

    float sme_error = 0.0f;
    float kupl_m16_error = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        sme_error = std::max(sme_error, std::abs(reference[i] - sme_scores[i]));
        kupl_m16_error = std::max(
            kupl_m16_error, std::abs(reference[i] - kupl_m16_scores[i]));
    }
    std::printf("max abs error: SME=%g KUPL-M16x4=%g\n",
        sme_error, kupl_m16_error);
    if (sme_error != 0.0f || kupl_m16_error != 0.0f) {
        std::puts("FAIL: page score mismatch");
        return 1;
    }

    for (int i = 0; i < 100; ++i) {
        pac_sme_page_scores_2x2(
            sme_packed_q.data(), sme_packed_k.data(), sme_scores.data());
        std::fill(kupl_m16_scores.begin(), kupl_m16_scores.end(), 0.0f);
        kupl_page_scores_m16x4(
            kupl_packed_q.data(), kupl_packed_k.data(), kupl_m16_scores.data());
    }

    const double sme_begin = now_ns();
    for (int i = 0; i < kIterations; ++i) {
        pac_sme_page_scores_2x2(
            sme_packed_q.data(), sme_packed_k.data(), sme_scores.data());
    }
    const double sme_ns = (now_ns() - sme_begin) / kIterations;

    std::fill(kupl_m16_scores.begin(), kupl_m16_scores.end(), 0.0f);
    const double kupl_m16_begin = now_ns();
    for (int i = 0; i < kIterations; ++i) {
        kupl_page_scores_m16x4(
            kupl_packed_q.data(), kupl_packed_k.data(), kupl_m16_scores.data());
    }
    const double kupl_m16_ns =
        (now_ns() - kupl_m16_begin) / kIterations;

    const double kupl_clear_begin = now_ns();
    for (int i = 0; i < kIterations; ++i) {
        std::fill(kupl_m16_scores.begin(), kupl_m16_scores.end(), 0.0f);
        kupl_page_scores_m16x4(
            kupl_packed_q.data(), kupl_packed_k.data(), kupl_m16_scores.data());
    }
    const double kupl_clear_ns =
        (now_ns() - kupl_clear_begin) / kIterations;

    constexpr double page_flops =
        2.0 * kHeads * kTokens * kDim;
    std::printf("SME 2x2: %.2f ns/page, %.2f GFLOPS\n",
        sme_ns, page_flops / sme_ns);
    std::printf("KUPL M16x4 accumulate: %.2f ns/page, %.2f GFLOPS\n",
        kupl_m16_ns, page_flops / kupl_m16_ns);
    std::printf("KUPL M16x4 clear+MMA:  %.2f ns/page, %.2f GFLOPS\n",
        kupl_clear_ns, page_flops / kupl_clear_ns);
    std::printf("M16x4 core speedup:    %.3fx\n", sme_ns / kupl_m16_ns);
    std::printf("M16x4 real speedup:    %.3fx\n", sme_ns / kupl_clear_ns);
    std::puts("PASS: KUPL MMA matches the contest page-score layout");
    return 0;
#endif
}
