#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <optional>
#include <limits>
#include <string>
#include <sstream>
#include <cmath>
#include <omp.h>

#include "Tensor.h"
#include "utils.h"
#include "testcase.h"

template <typename AccumType>
inline void ref_bf16_paged_mqa_logits(
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
    int64_t logits_size = batch_size * next_n * max_model_len;
    int64_t max_num_blocks = block_tables.sizes()[1];

    // 1. init output
    std::fill(output.data_ptr(), output.data_ptr() + logits_size, -INFINITY);

#pragma omp parallel for
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int64_t context_len = context_lens.data_ptr()[batch_idx];
        // 2.q context len set
        std::vector<int64_t> q_offsets(next_n);
        for (int64_t n = 0; n < next_n; ++n) {
            q_offsets[n] = context_len - next_n + n;
        }
        // 3. get k_cache
        int64_t num_blocks = ceil_div(context_len, block_size);
        int64_t total_tokens = num_blocks * block_size;

        std::vector<float> k_buffer(total_tokens * dim, -INFINITY);
        for (int64_t b = 0; b < num_blocks; ++b) {
            int64_t block_idx = block_tables.data_ptr()[batch_idx * max_num_blocks + b];
            if (block_idx < 0) {
                std::cout << " -> Invalid block, skipping\n";
                continue;
            }
            for (int64_t t = 0; t < block_size; ++t) {
                int64_t global_t = b * block_size + t;
                if (global_t >= total_tokens) break;
                int64_t kv_offset = ((block_idx * block_size + t)) * dim;
                for (int64_t d_idx = 0; d_idx < dim; ++d_idx) {
                    k_buffer[global_t * dim + d_idx] =
                        to_float(kv_cache.data_ptr()[kv_offset + d_idx]);
                }
            }
        }

        // 4. Q @ K^T
        int64_t weight_offset = batch_idx * next_n;
        auto weight_ptr = weights.data_ptr() + weight_offset * num_heads;
        for (int64_t n = 0; n < next_n; ++n) {
            for (int64_t h = 0; h < num_heads; ++h) {
                std::vector<float> q_vec(dim);
                int64_t q_offset = ((batch_idx * next_n + n) * num_heads + h) * dim;
                for (int64_t d_idx = 0; d_idx < dim; ++d_idx) {
                    q_vec[d_idx] = to_float(q.data_ptr()[q_offset + d_idx]);
                }

                // 5. scores mask
                std::vector<AccumType> scores(total_tokens, -INFINITY);
                for (int64_t t = 0; t < total_tokens; ++t) {
                    if (t > q_offsets[n] || t >= context_len) continue;

                    AccumType dot = 0.0f;
                    for (int64_t d_idx = 0; d_idx < dim; ++d_idx) {
                        dot += (AccumType)q_vec[d_idx] * (AccumType)k_buffer[t * dim + d_idx];
                    }
                    scores[t] = dot;
                }

                // 6. ReLU + weight
                float w = weight_ptr[n * num_heads + h];
                for (int64_t t = 0; t < total_tokens; ++t) {
                    if (scores[t] != -INFINITY) {
                        scores[t] = std::max<AccumType>(0, scores[t]) * (AccumType)w;
                    }
                }

                // 6. head reduce
                int64_t out_offset = (batch_idx * next_n + n) * max_model_len;
                for (int64_t t = 0; t < total_tokens; ++t) {
                    if (scores[t] != -INFINITY) {
                        if (output.data_ptr()[out_offset + t] == -INFINITY) {
                            output.data_ptr()[out_offset + t] = scores[t];
                        } else {
                            output.data_ptr()[out_offset + t] += scores[t];
                        }
                    }
                }
            }
        }

        for (int64_t n = 0; n < next_n; ++n) {
            int64_t out_offset = (batch_idx * next_n + n) * max_model_len;
            for (int64_t t = 0; t < max_model_len; ++t) {
                if (t > q_offsets[n] || t >= context_len) {
                    output.data_ptr()[out_offset + t] = -INFINITY;
                }
            }
        }
    }
}