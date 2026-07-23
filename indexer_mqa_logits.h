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
    // 当前是直接调用，优化时应该直接改为实际的内容
    ref_bf16_paged_mqa_logits<float>(q, kv_cache, block_tables, context_lens,
        weights, output, batch_size, next_n, num_heads, dim, block_size, max_model_len);

}