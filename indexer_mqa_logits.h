#pragma once

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include "ref_mqa_logits.h"
#include "allocator.h"
#include "Tensor.h"
#include "utils.h"

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#endif

#ifdef __aarch64__
extern "C" void pac_sme_tile_fused(
    const bfloat16_t *packed_q,
    const bfloat16_t *packed_k,
    const float *weights,
    float *output,
    int64_t valid_tokens);

asm(R"(
    .arch armv9-a+sme+sve2
    .text
    .align 4
    .global pac_sme_tile_fused
    .type pac_sme_tile_fused, %function
pac_sme_tile_fused:
    sub sp, sp, #192
    stp d8, d9, [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    ptrue p0.h
    ptrue p1.s

    zero {za}
    mov x5, x0
    mov x6, x1
    mov x7, #64

    /* BFMOPA: 64 kp iterations */
1:  ld1h {z0.h}, p0/z, [x5]
    ld1h {z1.h}, p0/z, [x5, #1, mul vl]
    ld1h {z2.h}, p0/z, [x5, #2, mul vl]
    ld1h {z3.h}, p0/z, [x5, #3, mul vl]
    ld1h {z4.h}, p0/z, [x6]
    bfmopa za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa za1.s, p0/m, p0/m, z1.h, z4.h
    bfmopa za2.s, p0/m, p0/m, z2.h, z4.h
    bfmopa za3.s, p0/m, p0/m, z3.h, z4.h
    add x5, x5, #256
    add x6, x6, #64
    subs x7, x7, #1
    b.ne 1b

    /* Post-processing: fuse ReLU + weight + reduce */
    eor z2.d, z2.d, z2.d                     /* zero for ReLU */
    eor z5.d, z5.d, z5.d                     /* result = 0 */

    add x11, sp, #128                        /* x11 = ZA row temp */
    mov x8, x2                               /* x8 = weights */

    /* Tile za0: heads 0..15 */
    mov w12, #0
2:  st1w {za0h.s[w12, 0]}, p1, [x11]        /* store head w12's scores */
    ld1w {z0.s}, p1/z, [x11]                /* load to SVE */
    fmax z0.s, p1/m, z0.s, z2.s             /* ReLU */
    add x14, x8, w12, uxtw #2               /* x14 = &weights[w12] */
    ldr w5, [x14]                            /* load weight */
    dup z3.s, w5                             /* broadcast to SVE */
    fmla z5.s, p1/m, z0.s, z3.s             /* z5[t] += score[t] × weight[w12] */
    add w12, w12, #1
    cmp w12, #16
    b.lo 2b

    /* Tile za1: heads 16..31 */
    mov w12, #0
3:  st1w {za1h.s[w12, 0]}, p1, [x11]
    ld1w {z0.s}, p1/z, [x11]
    fmax z0.s, p1/m, z0.s, z2.s
    add x14, x8, #64
    add x14, x14, w12, uxtw #2             /* x14 = &weights[16 + w12] */
    ldr w5, [x14]
    dup z3.s, w5
    fmla z5.s, p1/m, z0.s, z3.s
    add w12, w12, #1
    cmp w12, #16
    b.lo 3b

    /* Tile za2: heads 32..47 */
    mov w12, #0
4:  st1w {za2h.s[w12, 0]}, p1, [x11]
    ld1w {z0.s}, p1/z, [x11]
    fmax z0.s, p1/m, z0.s, z2.s
    add x14, x8, #128
    add x14, x14, w12, uxtw #2
    ldr w5, [x14]
    dup z3.s, w5
    fmla z5.s, p1/m, z0.s, z3.s
    add w12, w12, #1
    cmp w12, #16
    b.lo 4b

    /* Tile za3: heads 48..63 */
    mov w12, #0
5:  st1w {za3h.s[w12, 0]}, p1, [x11]
    ld1w {z0.s}, p1/z, [x11]
    fmax z0.s, p1/m, z0.s, z2.s
    add x14, x8, #192
    add x14, x14, w12, uxtw #2
    ldr w5, [x14]
    dup z3.s, w5
    fmla z5.s, p1/m, z0.s, z3.s
    add w12, w12, #1
    cmp w12, #16
    b.lo 5b

    /* Store result, masked by valid_tokens */
    whilelt p2.s, xzr, x4
    st1w {z5.s}, p2, [x3]

    smstop
    ldp d8, d9, [sp, #0]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    add sp, sp, #192
    ret
    .size pac_sme_tile_fused, .-pac_sme_tile_fused
)");
#endif

inline void indexer_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4> &q,          // [batch_size, next_n, num_heads, dim]
    const Tensor<bfloat16_t, 4> &kv_cache,   // [num_blocks, block_size, 1, dim]
    const Tensor<int64_t, 2> &block_tables,  // [batch_size, max_num_blocks]
    const Tensor<int64_t, 1> &context_lens,   // [batch_size]
    const Tensor<float, 2> &weights,         // [batch_size * next_n, num_heads]
    const Tensor<float, 2> &output,          // [batch_size * next_n, max_model_len] (output)
    int64_t batch_size,
    int64_t next_n,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len
) {
    const auto *q_ptr = q.data_ptr();
    const auto *kv_ptr = kv_cache.data_ptr();
    const auto *block_table_ptr = block_tables.data_ptr();
    const auto *context_len_ptr = context_lens.data_ptr();
    const auto *weight_ptr = weights.data_ptr();
    auto *output_ptr = output.data_ptr();
    const int64_t max_num_blocks = block_tables.size(1);

    FLASH_ASSERT(block_size == 64 && num_heads == 64 && dim == 128);
    FLASH_ASSERT(next_n > 0 && next_n <= 2);

#pragma omp parallel for schedule(static)
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t context_len = context_len_ptr[batch_idx];
        const int64_t num_blocks = ceil_div(context_len, block_size);

#ifdef __aarch64__
        alignas(64) bfloat16_t packed_q[2][64 * 128];
        alignas(64) bfloat16_t packed_k[64 * 128];
        const svbool_t pack_pg = svptrue_b32();
        const svuint32_t pack_offsets = svindex_u32(0, 256);

        for (int64_t n = 0; n < next_n; ++n) {
            const int64_t row = batch_idx * next_n + n;
            const bfloat16_t *q_row = q_ptr + row * num_heads * dim;
            for (int64_t kp = 0; kp < 64; ++kp) {
                for (int64_t hb = 0; hb < 4; ++hb) {
                    bfloat16_t *dst = packed_q[n] + (kp * 4 + hb) * 32;
                    for (int64_t h = 0; h < 16; ++h) {
                        const bfloat16_t *src =
                            q_row + (hb * 16 + h) * dim + kp * 2;
                        dst[h * 2] = src[0];
                        dst[h * 2 + 1] = src[1];
                    }
                }
            }
            float *out = output_ptr + row * max_model_len;
            std::fill(out, out + max_model_len, -INFINITY);
        }

        for (int64_t logical_block = 0; logical_block < num_blocks; ++logical_block) {
            const int64_t physical_block =
                block_table_ptr[batch_idx * max_num_blocks + logical_block];
            if (physical_block < 0) {
                continue;
            }
            const bfloat16_t *k_page =
                kv_ptr + physical_block * block_size * dim;

            for (int64_t tb = 0; tb < 4; ++tb) {
                for (int64_t kp = 0; kp < 64; ++kp) {
                    bfloat16_t *dst = packed_k + (tb * 64 + kp) * 32;
                    const bfloat16_t *src =
                        k_page + tb * 16 * dim + kp * 2;
                    const svuint32_t pairs = svld1_gather_u32offset_u32(
                        pack_pg, reinterpret_cast<const uint32_t *>(src),
                        pack_offsets);
                    svst1_u32(pack_pg, reinterpret_cast<uint32_t *>(dst), pairs);
                }
            }

            for (int64_t n = 0; n < next_n; ++n) {
                const int64_t row = batch_idx * next_n + n;
                const int64_t q_limit = context_len - next_n + n;
                const int64_t token_base = logical_block * block_size;
                const int64_t valid_tokens =
                    std::min<int64_t>(block_size, q_limit + 1 - token_base);
                if (valid_tokens <= 0) {
                    continue;
                }
                const int64_t token_tiles = ceil_div(valid_tokens, int64_t(16));
                const float *row_weights = weight_ptr + row * num_heads;
                float *out = output_ptr + row * max_model_len + token_base;
                for (int64_t tb = 0; tb < token_tiles; ++tb) {
                    const int64_t tile_valid =
                        std::min<int64_t>(16, valid_tokens - tb * 16);
                    pac_sme_tile_fused(
                        packed_q[n],
                        packed_k + tb * (64 * 32),
                        row_weights,
                        out + tb * 16,
                        tile_valid);
                }
            }
        }
#else
        for (int64_t n = 0; n < next_n; ++n) {
            const int64_t row = batch_idx * next_n + n;
            const int64_t q_limit = context_len - next_n + n;
            float *out = output_ptr + row * max_model_len;

            // Preserve the reference contract: every masked position is -inf.
            std::fill(out, out + max_model_len, -INFINITY);

            const bfloat16_t *q_row =
                q_ptr + row * num_heads * dim;
            const float *row_weights = weight_ptr + row * num_heads;

            for (int64_t logical_block = 0; logical_block < num_blocks; ++logical_block) {
                const int64_t token_base = logical_block * block_size;
                const int64_t valid_tokens = std::min<int64_t>(
                    block_size, q_limit + 1 - token_base);
                if (valid_tokens <= 0) {
                    break;
                }

                const int64_t physical_block =
                    block_table_ptr[batch_idx * max_num_blocks + logical_block];
                if (physical_block < 0) {
                    continue;
                }
                const bfloat16_t *k_page =
                    kv_ptr + physical_block * block_size * dim;

                alignas(64) float page_output[64] = {};

                // One 64-token KV page remains cache-resident while all heads
                // consume it. ReLU, weight and head reduction are fused into
                // page_output, so there is no per-head scores allocation.
                for (int64_t h = 0; h < num_heads; ++h) {
                    const bfloat16_t *q_head = q_row + h * dim;
                    const float weight = row_weights[h];

                    for (int64_t t = 0; t < valid_tokens; ++t) {
                        const bfloat16_t *k_token = k_page + t * dim;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < dim; ++d) {
                            dot += to_float(q_head[d]) * to_float(k_token[d]);
                        }
                        page_output[t] += std::max(0.0f, dot) * weight;
                    }
                }

                std::copy(page_output, page_output + valid_tokens, out + token_base);
            }
        }
#endif
    }
}
