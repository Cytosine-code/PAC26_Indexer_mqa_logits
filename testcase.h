#pragma once

#include <vector>
#include <optional>
#include <omp.h>
#include <random>
#include <algorithm>

#include "Tensor.h"
#include "utils.h"
#include "allocator.h"

struct TestCaseParams {
    int64_t batch_size;
    int64_t next_n;
    int64_t num_heads;
    int64_t dim;

    int64_t block_size;
    int64_t max_model_len;

    bool check_correctness;
    int64_t num_runs;
    int64_t avg_kv_len;
};

struct TestCaseData {
    Tensor<bfloat16_t, 4> q;          // [batch_size, next_n, num_heads, dim]
    Tensor<bfloat16_t, 4> kv_cache;  // [total_num_blocks, block_size, 1, dim]
    Tensor<int64_t, 2> block_tables;  // [batch_size, max_num_blocks]
    Tensor<int64_t, 1> context_lens;  // [batch_size]
    Tensor<float, 2> weights;        // [batch_size * next_n, num_heads]

    Tensor<float, 2> logits;         // [batch_size * next_n, max_model_len]
    Tensor<float, 2> ref_logits;     // [batch_size * next_n, max_model_len]

    int64_t total_context_len;
};

inline int64_t calculate_testcase_data_size(const TestCaseParams &p)
{
    int64_t total_size = 0;
    int64_t max_num_blocks = ceil_div(p.max_model_len, p.block_size);
    int64_t total_num_blocks = p.batch_size * max_num_blocks;
    total_size += sizeof(bfloat16_t) * p.batch_size * p.next_n * p.num_heads * p.dim;              // q
    total_size += sizeof(bfloat16_t) * total_num_blocks * p.block_size * p.dim;                   // kv_cache
    total_size += sizeof(int64_t) * p.batch_size * max_num_blocks;                                // block_tables
    total_size += sizeof(int64_t) * p.batch_size;                                                  // context_lens
    total_size += sizeof(float) * p.batch_size * p.next_n * p.num_heads;                            // weights
    total_size += sizeof(float) * p.batch_size * p.next_n * p.max_model_len;                        // logits
    total_size += sizeof(float) * p.batch_size * p.next_n * p.max_model_len;                        // ref_logits
    return total_size + DEFAULT_ALIGNMENT * 7;
}

inline TestCaseData generate_testcase_data(const TestCaseParams &p, Allocator &allocator, std::mt19937 &rng)
{
    auto gen_bf16 = [&]() -> bfloat16_t {
        std::normal_distribution<> distr;
        return to_bf16(std::min(std::max(-3.0, distr(rng)), 3.0) / 3);
    };
    auto gen_float = [&]() -> float {
        std::normal_distribution<> distr;
        return std::min(std::max(-3.0, distr(rng)), 3.0) / 3;
    };

    int64_t max_num_blocks = ceil_div(p.max_model_len, p.block_size);
    int64_t total_num_blocks = p.batch_size * max_num_blocks;

    auto q = generate_tensor<bfloat16_t, 4>(allocator, {p.batch_size, p.next_n, p.num_heads, p.dim}, gen_bf16);
    auto kv_cache = create_tensor<bfloat16_t, 4>(allocator, {total_num_blocks, p.block_size, 1, p.dim});

    std::vector<std::mt19937> block_rng_list;
    for (int64_t i = 0; i < total_num_blocks; ++i) {
        std::mt19937 block_rng(rng());
        block_rng_list.push_back(block_rng);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < total_num_blocks; ++i) {
        std::normal_distribution<> distr;
        auto &block_rng = block_rng_list[i];
        for (int64_t j = 0; j < p.block_size * p.dim; ++j) {
            kv_cache.data_ptr()[i * p.block_size * p.dim + j] = to_bf16(std::min(std::max(-3.0, distr(block_rng)), 3.0) / 3);
        }
    }

    auto weights = generate_tensor<float, 2>(allocator, {p.batch_size * p.next_n, p.num_heads}, gen_float);

    int64_t total_context_len = 0;
    auto context_lens = generate_tensor<int64_t, 1>(allocator, {p.batch_size}, [&] {
        float f = 1 + gen_float() * 0.3;
        int64_t len = static_cast<int64_t>(p.avg_kv_len * f);
        len = std::min(std::max(p.next_n, len), p.max_model_len);
        total_context_len += len;
        return len;
    });

    auto block_tables = create_tensor<int64_t, 2>(allocator, {p.batch_size, max_num_blocks});
    for (int i = 0; i < total_num_blocks; ++i) {
        block_tables.data_ptr()[i] = i;
    }
    std::shuffle(block_tables.data_ptr(), block_tables.data_ptr() + total_num_blocks, rng);

    auto logits = create_tensor<float, 2>(allocator, {p.batch_size * p.next_n, p.max_model_len});
    auto ref_logits = create_tensor<float, 2>(allocator, {p.batch_size * p.next_n, p.max_model_len});

    return {q, kv_cache, block_tables, context_lens, weights, logits, ref_logits, total_context_len};
}

inline std::vector<TestCaseData> generate_datalist(const TestCaseParams &p, Allocator &allocator, std::mt19937 &rng)
{
    std::vector<TestCaseData> data_list;
    int64_t single_size = calculate_testcase_data_size(p);
    while (allocator.remain() >= single_size && data_list.size() < p.num_runs) {
        TestCaseData data = generate_testcase_data(p, allocator, rng);
        data_list.push_back(data);
    }
    return data_list;
}