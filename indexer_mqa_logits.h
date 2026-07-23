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

    // The contest shapes use 64-token pages. Keeping one page of partial
    // results on the stack avoids a context-length-sized temporary buffer.
    FLASH_ASSERT(block_size <= 64);

#pragma omp parallel for schedule(static)
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t context_len = context_len_ptr[batch_idx];
        const int64_t num_blocks = ceil_div(context_len, block_size);

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

#ifdef __aarch64__
                    // Load this head's fixed 128-element query once, then reuse
                    // the four registers for every token in the current page.
                    const svbool_t pg16 = svptrue_b16();
                    const svbool_t pg32 = svptrue_b32();
                    const svbfloat16_t q0 = svld1_bf16(pg16, q_head);
                    const svbfloat16_t q1 = svld1_bf16(pg16, q_head + 32);
                    const svbfloat16_t q2 = svld1_bf16(pg16, q_head + 64);
                    const svbfloat16_t q3 = svld1_bf16(pg16, q_head + 96);
#endif
                    for (int64_t t = 0; t < valid_tokens; ++t) {
                        const bfloat16_t *k_token = k_page + t * dim;
#ifdef __aarch64__
                        svfloat32_t acc = svdup_f32(0.0f);
                        acc = svbfdot_f32(acc, q0, svld1_bf16(pg16, k_token));
                        acc = svbfdot_f32(acc, q1, svld1_bf16(pg16, k_token + 32));
                        acc = svbfdot_f32(acc, q2, svld1_bf16(pg16, k_token + 64));
                        acc = svbfdot_f32(acc, q3, svld1_bf16(pg16, k_token + 96));
                        const float dot = svaddv_f32(pg32, acc);
#else
                        float dot = 0.0f;
                        for (int64_t d = 0; d < dim; ++d) {
                            dot += to_float(q_head[d]) * to_float(k_token[d]);
                        }
#endif
                        page_output[t] += std::max(0.0f, dot) * weight;
                    }
                }

                std::copy(page_output, page_output + valid_tokens, out + token_base);
            }
        }
    }
}
